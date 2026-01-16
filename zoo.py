import logging
import os
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image
import json

import fiftyone as fo
from fiftyone import Model
from fiftyone.core.models import SamplesMixin, SupportsGetItem, TorchModelMixin
from fiftyone.core.labels import Classification, Classifications
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

# Utility functions
def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MedGemmaGetItem(GetItem):
    """Transform for loading images and extracting prompts for MedGemma.
    
    This class handles data loading in DataLoader worker processes:
    - Loads images from filepaths
    - Extracts per-sample prompts if configured
    """
    
    def __init__(self, field_mapping=None, prompt_field=None):
        """Initialize the GetItem transform.
        
        Args:
            field_mapping: Optional dict mapping required_keys to dataset fields
            prompt_field: Name of the field containing per-sample prompts
        """
        self.prompt_field = prompt_field
        super().__init__(field_mapping=field_mapping)
    
    @property
    def required_keys(self):
        """Return list of fields needed from each sample."""
        keys = ["filepath"]
        if self.prompt_field is not None:
            keys.append("prompt_field")
        return keys
    
    def __call__(self, sample_dict):
        """Load and preprocess a single sample.
        
        Args:
            sample_dict: Dict with keys from required_keys
        
        Returns:
            Dict containing:
                - "image": PIL Image loaded from filepath
                - "prompt": Per-sample prompt (or None if not configured)
        """
        filepath = sample_dict["filepath"]
        
        # Load image (runs in parallel via DataLoader workers)
        image = Image.open(filepath).convert("RGB")
        
        # Extract prompt from sample if available
        prompt = None
        if "prompt_field" in sample_dict:
            prompt = sample_dict["prompt_field"]
        
        return {"image": image, "prompt": prompt}


class medgemma(Model, SamplesMixin, SupportsGetItem, TorchModelMixin):
    """A FiftyOne model for running MedGemma vision tasks with batching support.
    
    Inheritance:
        - Model: Base class with required interface
        - SamplesMixin: Enables per-sample field access (for prompts)
        - SupportsGetItem: Enables DataLoader-based batching
        - TorchModelMixin: Provides custom collation for variable-size inputs
    """

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        quantized: bool = None,
        **kwargs
    ):
        # Initialize mixins
        SamplesMixin.__init__(self)
        SupportsGetItem.__init__(self)
        
        # Required for SamplesMixin - tracks fields needed from samples
        self._fields = {}
        
        # Required for batching - preprocessing is handled by GetItem
        self._preprocess = False
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt
        self._operation = operation
        self.prompt = prompt  # Global default prompt
        self.quantized = quantized
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Base model kwargs that are always needed
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device,
        }
        
        # Set optimizations based on CUDA device capabilities
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self.device)
            # Enable bfloat16 on Ampere+ GPUs (compute capability 8.0+)
            if capability[0] >= 8:
                model_kwargs["torch_dtype"] = torch.bfloat16
            
            # Only apply quantization if device is CUDA
            if self.quantized:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif self.quantized:
            logger.warning("Quantization is only supported on CUDA devices. Ignoring quantization request.")
            
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        logger.info("Loading processor")

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )

        self.model.eval()

    # ============ FROM Model BASE CLASS ============

    @property
    def media_type(self):
        """The media type processed by the model."""
        return "image"

    @property
    def transforms(self):
        """Preprocessing transforms applied to inputs.
        
        For SupportsGetItem models, preprocessing happens in GetItem,
        so return None.
        """
        return None

    @property
    def preprocess(self):
        """Whether model should apply preprocessing."""
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value):
        """Allow FiftyOne to control preprocessing."""
        self._preprocess = value

    @property
    def ragged_batches(self):
        """Whether this model supports batches with varying sizes.
        
        MUST return False to enable batching, even if your inputs
        have variable sizes! Handle variable sizes via custom collate_fn.
        """
        return False

    # ============ FROM SamplesMixin ============

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields

    def _get_field(self):
        """Get the prompt field name from needs_fields."""
        if "prompt_field" in self.needs_fields:
            return self.needs_fields["prompt_field"]
        return next(iter(self.needs_fields.values()), None)

    # ============ FROM TorchModelMixin ============

    @property
    def has_collate_fn(self):
        """Whether this model provides custom batch collation.
        
        Return True for variable-size inputs (images of different dimensions).
        """
        return True

    @property
    def collate_fn(self):
        """Custom collation function for batching.
        
        Returns items as a list without stacking, allowing variable-size images.
        """
        @staticmethod
        def identity_collate(batch):
            """Return list of items without stacking."""
            return batch
        return identity_collate

    # ============ FROM SupportsGetItem ============

    def build_get_item(self, field_mapping=None):
        """Build the GetItem transform for data loading.
        
        Args:
            field_mapping: Optional dict mapping required_keys to dataset fields
        
        Returns:
            MedGemmaGetItem instance configured for this model
        """
        # Extract prompt field name from field_mapping or _fields
        prompt_field = None
        if field_mapping and "prompt_field" in field_mapping:
            prompt_field = field_mapping["prompt_field"]
        elif "prompt_field" in self._fields:
            prompt_field = self._fields["prompt_field"]
        
        return MedGemmaGetItem(
            field_mapping=field_mapping,
            prompt_field=prompt_field
        )

    # ============ OPERATION AND PROMPT PROPERTIES ============

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in MEDGEMMA_OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(MEDGEMMA_OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        """Return custom system prompt if set, otherwise return default for current operation."""
        return self._custom_system_prompt if self._custom_system_prompt is not None else MEDGEMMA_OPERATIONS[self.operation]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    # ============ HELPER METHODS ============

    def _parse_json(self, s: str) -> Optional[Dict]:
        """Parse JSON from model output.
        
        Args:
            s: String output from the model to parse
            
        Returns:
            Dict: Parsed JSON dictionary if successful
            None: If parsing fails or input is invalid
            Original input: If input is not a string
        """
        if not isinstance(s, str):
            return s
            
        # Handle JSON wrapped in markdown code blocks
        if "```json" in s:
            try:
                s = s.split("```json")[1].split("```")[0].strip()
            except IndexError:
                logger.debug("Failed to extract JSON from markdown blocks")
                return None
        
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e}. First 200 chars: {s[:200]}")
            return None

    def _to_classifications(self, data: Dict) -> fo.Classifications:
        """Convert JSON classification data to FiftyOne Classifications.
        
        Args:
            data: Dictionary containing a 'classifications' list where each item has:
                - 'label': String class label
            
        Returns:
            fo.Classifications object containing the converted classification annotations
            
        Example input:
            {
                "classifications": [
                    {"label": "condition_1"},
                    {"label": "condition_2"}
                ]
            }
        """
        classifications = []
        
        try:
            # Extract the classifications list from the input dictionary
            classes = data.get("classifications", [])
            
            # Process each classification dictionary
            for cls in classes:
                try:
                    if not isinstance(cls, dict) or "label" not in cls:
                        logger.debug(f"Invalid classification format: {cls}")
                        continue
                        
                    classification = fo.Classification(
                        label=str(cls["label"]),
                    )
                    classifications.append(classification)

                except Exception as e:
                    logger.debug(f"Error processing classification {cls}: {e}")
                    continue

        except Exception as e:
            logger.debug(f"Error processing classifications data: {e}")
            
        return fo.Classifications(classifications=classifications)

    def _run_single_inference(self, image: Image.Image, prompt: str) -> str:
        """Run inference on a single image with a prompt.
        
        Args:
            image: PIL Image to process
            prompt: Text prompt for the model
            
        Returns:
            str: Raw text output from the model
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        
        # Move tensors to the device
        if self.device == "cuda":
            text = {k: v.to(self.device, dtype=torch.bfloat16) for k, v in text.items()}
        else:
            text = {k: v.to(self.device) for k, v in text.items()}

        input_len = text["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **text, 
                max_new_tokens=8192, 
                do_sample=False
            )
            generation = generation[0][input_len:]

        output_text = self.processor.decode(generation, skip_special_tokens=True)
        return output_text

    def _format_output(self, output_text: str) -> Union[fo.Classifications, str]:
        """Format model output based on operation type.
        
        Args:
            output_text: Raw text output from the model
            
        Returns:
            fo.Classifications for classify operation, str for VQA
        """
        if self.operation == "vqa":
            return output_text.strip()
        elif self.operation == "classify":
            parsed_output = self._parse_json(output_text)
            return self._to_classifications(parsed_output)
        else:
            return output_text.strip()

    # ============ PREDICTION METHODS ============

    def predict_all(self, batch, samples=None):
        """Process a batch of images through the model.
        
        This is the main batching method called by FiftyOne when using DataLoader.
        
        Args:
            batch: List of dicts from GetItem, each containing:
                - "image": PIL Image
                - "prompt": Per-sample prompt (or None)
            samples: Optional list of FiftyOne samples (for metadata access)
        
        Returns:
            List of predictions (fo.Classifications for classify, str for VQA)
        """
        if not batch:
            return []
        
        results = []
        
        for i, item in enumerate(batch):
            # Handle both dict format (from GetItem) and direct PIL Image
            if isinstance(item, dict):
                image = item.get("image")
                prompt_from_batch = item.get("prompt")
            else:
                # Direct PIL Image passed (e.g., from predict())
                image = item
                prompt_from_batch = None
            
            # Resolve prompt: per-sample > global default
            resolved_prompt = (
                prompt_from_batch 
                if prompt_from_batch is not None 
                else self.prompt
            )
            
            if resolved_prompt is None:
                raise ValueError(
                    "No prompt provided. Either set a global prompt via model.prompt "
                    "or use prompt_field parameter in apply_model()."
                )
            
            # Run inference
            output_text = self._run_single_inference(image, resolved_prompt)
            
            # Format output based on operation
            result = self._format_output(output_text)
            results.append(result)
        
        return results

    def predict(self, arg, sample=None):
        """Process a single image with the model.
        
        This method is called in non-batched mode or as a fallback.
        
        Args:
            arg: PIL Image, numpy array, or dict from GetItem
            sample: Optional FiftyOne sample (passed by FiftyOne when using SamplesMixin)
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        # Handle different input formats
        if isinstance(arg, dict):
            # Already from GetItem (batched mode fallback)
            batch_item = arg
        elif isinstance(arg, np.ndarray):
            # Convert numpy to PIL
            image = Image.fromarray(arg)
            
            # Extract prompt from sample if available (SamplesMixin path)
            prompt = None
            if sample is not None and "prompt_field" in self._fields:
                field_name = self._fields["prompt_field"]
                if sample.has_field(field_name):
                    prompt = sample.get_field(field_name)
            
            batch_item = {"image": image, "prompt": prompt}
        else:
            # Assume PIL Image
            image = arg
            
            # Extract prompt from sample if available (SamplesMixin path)
            prompt = None
            if sample is not None and "prompt_field" in self._fields:
                field_name = self._fields["prompt_field"]
                if sample.has_field(field_name):
                    prompt = sample.get_field(field_name)
            
            batch_item = {"image": image, "prompt": prompt}
        
        # Call batch method with single item
        results = self.predict_all([batch_item], samples=[sample] if sample else None)
        return results[0]
