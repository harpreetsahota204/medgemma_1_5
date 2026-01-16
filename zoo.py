import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
import json

import fiftyone as fo
from fiftyone import Model
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
from fiftyone.utils.torch import GetItem

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """You are an expert radiologist, histopathologist, ophthalmologist, and dermatologist.

Your expert opinion is needed for in classifiying medical images. A given image may have multiple relevant classifications.  

You may be requested to select from a list of classifications, or asked to leverage your expertise medical for a diagnosis.

In any event, report your classifications as JSON array in this format: 

```json
{
    "classifications": [
        {
            "label": "descriptive medical condition or relevant label",
            "label": "descriptive medical condition or relevant label",
            ...,
        }
    ]
}
```

Always return your response as valid JSON wrapped in ```json blocks.  You may produce multiple lables if they are relevant or if you are asked to. Do not report your confidence.
"""

DEFAULT_VQA_SYSTEM_PROMPT = """You are an expert radiologist, histopathologist, ophthalmologist, and dermatologist. You are asked to provide leverage your expertise to answers to medical questions.

You may be provided with a simple query, patient history with a complex query, asked to provide a medical diagnosis, or any variety of medical question.
"""

MEDGEMMA_OPERATIONS = {
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT
}

logger = logging.getLogger(__name__)


def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MedGemmaGetItem(GetItem):
    """GetItem transform for loading images and extracting per-sample prompts."""
    
    def __init__(self, field_mapping=None, prompt_field=None):
        self.prompt_field = prompt_field
        super().__init__(field_mapping=field_mapping)
    
    @property
    def required_keys(self):
        keys = ["filepath"]
        if self.prompt_field:
            keys.append(self.prompt_field)
        return keys
    
    def __call__(self, sample_dict):
        image = Image.open(sample_dict["filepath"]).convert("RGB")
        prompt = sample_dict.get(self.prompt_field) if self.prompt_field else None
        return {"image": image, "prompt": prompt}


class medgemma(Model, SupportsGetItem, TorchModelMixin):
    """A FiftyOne model for running MedGemma vision tasks with batching support."""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        quantized: bool = None,
        max_new_tokens: int = 8192,
        **kwargs
    ):
        SupportsGetItem.__init__(self)
        
        self._preprocess = False
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt
        self._operation = operation
        self.prompt = prompt
        self.quantized = quantized
        self.max_new_tokens = max_new_tokens
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device,
        }
        
        self._inference_dtype = None
        
        if self.device == "cuda":
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:
                model_kwargs["torch_dtype"] = torch.bfloat16
                self._inference_dtype = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float16
                self._inference_dtype = torch.float16
            
            if self.quantized:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        self.model.eval()

    # =========================================================================
    # Context manager
    # =========================================================================
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False

    # =========================================================================
    # Model base class requirements
    # =========================================================================

    @property
    def media_type(self):
        return "image"

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self):
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value):
        self._preprocess = value

    @property
    def ragged_batches(self):
        return False

    # =========================================================================
    # Per-sample prompt field support
    # =========================================================================

    @property
    def needs_fields(self):
        return self._fields
    
    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields

    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            return self.needs_fields["prompt_field"]
        return next(iter(self.needs_fields.values()), None)

    # =========================================================================
    # TorchModelMixin requirements
    # =========================================================================

    @property
    def has_collate_fn(self):
        return True

    @property
    def collate_fn(self):
        return lambda batch: batch

    # =========================================================================
    # SupportsGetItem requirements
    # =========================================================================

    def get_item(self):
        return MedGemmaGetItem(prompt_field=self._get_field())

    def build_get_item(self, field_mapping=None):
        return MedGemmaGetItem(field_mapping=field_mapping, prompt_field=self._get_field())

    # =========================================================================
    # Operation and prompt properties
    # =========================================================================

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        self._operation = value

    @property
    def system_prompt(self):
        if self._custom_system_prompt is not None:
            return self._custom_system_prompt
        return MEDGEMMA_OPERATIONS.get(self.operation, DEFAULT_VQA_SYSTEM_PROMPT)

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    # =========================================================================
    # Output formatting
    # =========================================================================

    def _parse_json(self, s: str) -> Optional[Dict]:
        """Parse JSON from model output, handling markdown code blocks."""
        try:
            if "```json" in s:
                s = s.split("```json")[1].split("```")[0].strip()
            return json.loads(s)
        except (json.JSONDecodeError, IndexError):
            logger.debug(f"Failed to parse JSON from: {s[:200]}")
            return None

    def _to_classifications(self, data: Optional[Dict]) -> fo.Classifications:
        """Convert parsed JSON to FiftyOne Classifications."""
        if data is None:
            return fo.Classifications(classifications=[])
        
        classifications = []
        for cls in data.get("classifications", []):
            if isinstance(cls, dict) and "label" in cls:
                classifications.append(fo.Classification(label=str(cls["label"])))
        return fo.Classifications(classifications=classifications)

    def _format_output(self, output_text: str) -> Union[fo.Classifications, str]:
        """Format model output based on operation type."""
        if self.operation == "classify":
            return self._to_classifications(self._parse_json(output_text))
        return output_text.strip()

    # =========================================================================
    # Prediction methods
    # =========================================================================

    def predict_all(self, batch, preprocess=None):
        """Batch prediction with true batching via processor padding."""
        if not batch:
            return []
        
        # Extract images and prompts
        images = []
        prompts = []
        for item in batch:
            if isinstance(item, dict):
                images.append(item["image"])
                prompts.append(item.get("prompt") or self.prompt)
            else:
                images.append(item)
                prompts.append(self.prompt)
        
        # Build messages for each image
        all_messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}
            ]
            for img, prompt in zip(images, prompts)
        ]
        
        # Apply chat template (tokenize=False for batching)
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in all_messages
        ]
        
        # Batch process with padding
        # Gemma3 processor expects images as list of lists (one list per text entry)
        batched_images = [[img] for img in images]
        inputs = self.processor(text=texts, images=batched_images, padding=True, return_tensors="pt")
        
        if self._inference_dtype:
            inputs = inputs.to(self.model.device, dtype=self._inference_dtype)
        else:
            inputs = inputs.to(self.model.device)
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode generated tokens only
        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):]
            for i in range(len(images))
        ]
        
        output_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        return [self._format_output(text) for text in output_texts]

    def _predict(self, image: Image.Image, prompt: str) -> Union[fo.Classifications, str]:
        """Single image inference."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True
        )
        
        if self._inference_dtype:
            inputs = inputs.to(self.model.device, dtype=self._inference_dtype)
        else:
            inputs = inputs.to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens, 
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]

        output_text = self.processor.decode(generation, skip_special_tokens=True)
        return self._format_output(output_text)

    def predict(self, image, sample=None):
        """Process an image with the model."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        prompt = self.prompt
        if sample is not None and self._get_field():
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)
        
        return self._predict(image, prompt)
