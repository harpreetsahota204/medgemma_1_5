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

DEFAULT_DETECTION_SYSTEM_PROMPT = """Instructions:
The following user query will require outputting bounding boxes. The format of bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left corner and (y1, x1) the bottom-right corner. This implies that x0 < x1 and y0 < y1. Always normalize the x and y coordinates the range [0, 1000], meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150. You MUST output a single parseable json list of objects enclosed into ```json...``` brackets, for instance ```json[{"box_2d": [800, 3, 840, 471], "label": "car"}, {"box_2d": [400, 22, 600, 73], "label": "dog"}]``` is a valid output. Now answer to the user query.

Remember "left" refers to the patient's left side where the heart is and sometimes underneath an L in the upper right corner of the image.
"""

MEDGEMMA_OPERATIONS = {
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    "detect": DEFAULT_DETECTION_SYSTEM_PROMPT
}

logger = logging.getLogger(__name__)


def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pad_image_to_square(image: Image.Image) -> Image.Image:
    """Pad image to square to preserve aspect ratio during model resize.
    
    Required for detection tasks where bounding box coordinates must map
    correctly between model output (normalized 0-1000) and original image.
    """
    image_array = np.array(image)
    
    # Handle grayscale
    if len(image_array.shape) < 3:
        image_array = np.stack([image_array] * 3, axis=-1)
    # Handle RGBA
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    
    h, w = image_array.shape[:2]
    
    if h < w:
        dh = w - h
        image_array = np.pad(
            image_array, ((dh // 2, dh - dh // 2), (0, 0), (0, 0))
        )
    elif w < h:
        dw = h - w
        image_array = np.pad(
            image_array, ((0, 0), (dw // 2, dw - dw // 2), (0, 0))
        )
    
    return Image.fromarray(image_array)


class MedGemmaGetItem(GetItem):
    """GetItem transform for loading images and extracting per-sample prompts."""
    
    def __init__(self, field_mapping=None, use_prompt_field=False, operation=None):
        # Set before super().__init__() since it accesses required_keys
        self.use_prompt_field = use_prompt_field
        self.operation = operation
        super().__init__(field_mapping=field_mapping)
    
    @property
    def required_keys(self):
        keys = ["filepath"]
        if self.use_prompt_field:
            keys.append("prompt_field")  # Logical key - FiftyOne maps this to actual field
        return keys
    
    def __call__(self, sample_dict):
        image = Image.open(sample_dict["filepath"]).convert("RGB")
        
        # Square padding for detection to preserve aspect ratio
        if self.operation == "detect":
            image = pad_image_to_square(image)
        
        prompt = sample_dict.get("prompt_field")  # Access via logical key
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
        return MedGemmaGetItem(
            use_prompt_field=self._get_field() is not None,
            operation=self.operation
        )

    def build_get_item(self, field_mapping=None):
        # Check if prompt_field is being used (either from needs_fields or field_mapping)
        use_prompt = self._get_field() is not None or (field_mapping and "prompt_field" in field_mapping)
        return MedGemmaGetItem(
            field_mapping=field_mapping,
            use_prompt_field=use_prompt,
            operation=self.operation
        )

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

    def _to_classifications(self, data) -> fo.Classifications:
        """Convert parsed JSON to FiftyOne Classifications."""
        if data is None:
            return fo.Classifications(classifications=[])
        
        # Handle both formats:
        # - {"classifications": [{"label": "..."}]}
        # - [{"label": "..."}]
        if isinstance(data, list):
            items = data
        else:
            items = data.get("classifications", [])
        
        classifications = []
        for cls in items:
            if isinstance(cls, dict) and "label" in cls:
                classifications.append(fo.Classification(label=str(cls["label"])))
        return fo.Classifications(classifications=classifications)

    def _to_detections(self, data) -> fo.Detections:
        """Convert parsed JSON to FiftyOne Detections.
        
        Model outputs bounding boxes as [y0, x0, y1, x1] normalized to [0, 1000].
        FiftyOne expects [x, y, width, height] normalized to [0, 1].
        """
        if data is None:
            return fo.Detections(detections=[])
        
        # Handle both list format and wrapped format
        if isinstance(data, list):
            items = data
        else:
            items = data.get("detections", [])
        
        detections = []
        for item in items:
            if not isinstance(item, dict):
                continue
            
            box_2d = item.get("box_2d")
            label = item.get("label", "object")
            
            if box_2d and len(box_2d) == 4:
                # Model format: [y0, x0, y1, x1] normalized to 0-1000
                y0, x0, y1, x1 = box_2d
                
                # Convert to FiftyOne format: [x, y, width, height] normalized to 0-1
                x = x0 / 1000.0
                y = y0 / 1000.0
                width = (x1 - x0) / 1000.0
                height = (y1 - y0) / 1000.0
                
                detection = fo.Detection(
                    label=str(label),
                    bounding_box=[x, y, width, height]
                )
                detections.append(detection)
        
        return fo.Detections(detections=detections)

    def _format_output(self, output_text: str) -> Union[fo.Classifications, fo.Detections, str]:
        """Format model output based on operation type."""
        if self.operation == "classify":
            return self._to_classifications(self._parse_json(output_text))
        elif self.operation == "detect":
            return self._to_detections(self._parse_json(output_text))
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

    def _predict(self, image: Image.Image, prompt: str) -> Union[fo.Classifications, fo.Detections, str]:
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
        
        # Square padding for detection to preserve aspect ratio
        if self.operation == "detect":
            image = pad_image_to_square(image)
        
        prompt = self.prompt
        if sample is not None and self._get_field():
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)
        
        return self._predict(image, prompt)
