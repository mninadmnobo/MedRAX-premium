from typing import Any, Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field

import torch

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from PIL import Image

from transformers import (
    BertTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    GenerationConfig,
)


class ChestXRayInput(BaseModel):
    """Input for chest X-ray analysis tools. Only supports JPG or PNG images."""

    image_path: str = Field(
        ..., description="Path to the radiology image file, only supports JPG or PNG images"
    )


class ChestXRayReportGeneratorTool(BaseTool):
    """Tool that generates comprehensive chest X-ray reports with both findings and impressions.

    This tool uses two Vision-Encoder-Decoder models (ViT-BERT) trained on CheXpert
    and MIMIC-CXR datasets to generate structured radiology reports. It automatically
    generates both detailed findings and impression summaries for each chest X-ray,
    following standard radiological reporting format.

    The tool uses:
    - Findings model: Generates detailed observations of all visible structures
    - Impression model: Provides concise clinical interpretation and key diagnoses
    """

    name: str = "chest_xray_report_generator"
    description: str = (
        "A tool that analyzes chest X-ray images and generates comprehensive radiology reports "
        "containing both detailed findings and impression summaries. Input should be the path "
        "to a chest X-ray image file. Output is a structured report with both detailed "
        "observations and key clinical conclusions."
    )
    device: Optional[str] = "cuda"
    args_schema: Type[BaseModel] = ChestXRayInput
    findings_model: VisionEncoderDecoderModel = None
    impression_model: VisionEncoderDecoderModel = None
    findings_tokenizer: BertTokenizer = None
    impression_tokenizer: BertTokenizer = None
    findings_processor: ViTImageProcessor = None
    impression_processor: ViTImageProcessor = None
    generation_args: Dict[str, Any] = None

    def __init__(
        self,
        cache_dir: str = "/model-weights",
        device: Optional[str] = "cuda",
        findings_model_path: str = "IAMJB/chexpert-mimic-cxr-findings-baseline",
        impression_model_path: str = "IAMJB/chexpert-mimic-cxr-impression-baseline",
    ):
        """Initialize the ChestXRayReportGeneratorTool with both findings and impression models.

        Args:
            cache_dir: HF cache directory (used when model paths are HF repo IDs).
            device: Device to run models on.
            findings_model_path: HF repo ID or local path to the findings model.
            impression_model_path: HF repo ID or local path to the impression model.
        """
        super().__init__()
        self.device = torch.device(device) if device else "cuda"

        # Initialize findings model
        self.findings_model = VisionEncoderDecoderModel.from_pretrained(
            findings_model_path, cache_dir=cache_dir, torch_dtype=torch.float16
        ).eval()
        self.findings_tokenizer = BertTokenizer.from_pretrained(
            findings_model_path, cache_dir=cache_dir
        )
        self.findings_processor = ViTImageProcessor.from_pretrained(
            findings_model_path, cache_dir=cache_dir
        )

        # Initialize impression model
        self.impression_model = VisionEncoderDecoderModel.from_pretrained(
            impression_model_path, cache_dir=cache_dir, torch_dtype=torch.float16
        ).eval()
        self.impression_tokenizer = BertTokenizer.from_pretrained(
            impression_model_path, cache_dir=cache_dir
        )
        self.impression_processor = ViTImageProcessor.from_pretrained(
            impression_model_path, cache_dir=cache_dir
        )

        # Move models to device
        self.findings_model = self.findings_model.to(self.device)
        self.impression_model = self.impression_model.to(self.device)

        # Default generation arguments
        self.generation_args = {
            "num_return_sequences": 1,
            "max_length": 128,
            "use_cache": True,
            "beam_width": 2,
        }

    def _process_image(
        self, image_path: str, processor: ViTImageProcessor, model: VisionEncoderDecoderModel
    ) -> torch.Tensor:
        """Process the input image for a specific model.

        Args:
            image_path (str): Path to the input image.
            processor: Image processor for the specific model.
            model: The model to process the image for.

        Returns:
            torch.Tensor: Processed image tensor ready for model input.
        """
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values

        expected_size = model.config.encoder.image_size
        actual_size = pixel_values.shape[-1]

        if expected_size != actual_size:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                size=(expected_size, expected_size),
                mode="bilinear",
                align_corners=False,
            )

        pixel_values = pixel_values.to(self.device)

        return pixel_values

    def _generate_report_section(
        self, pixel_values: torch.Tensor, model: VisionEncoderDecoderModel, tokenizer: BertTokenizer
    ) -> str:
        """Generate a report section using the specified model.

        Args:
            pixel_values: Processed image tensor.
            model: The model to use for generation.
            tokenizer: The tokenizer for the model.

        Returns:
            str: Generated text for the report section.
        """
        generation_config = GenerationConfig(
            **{
                **self.generation_args,
                "bos_token_id": model.config.bos_token_id,
                "eos_token_id": model.config.eos_token_id,
                "pad_token_id": model.config.pad_token_id,                "decoder_start_token_id": tokenizer.cls_token_id,
            }
        )

        generated_ids = model.generate(pixel_values, generation_config=generation_config)

        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _generate_report_section_sampled(
        self, pixel_values: torch.Tensor, model: VisionEncoderDecoderModel, 
        tokenizer: BertTokenizer, temperature: float = 1.0
    ) -> str:
        """Generate a report section with sampling for diversity.

        Args:
            pixel_values: Processed image tensor.
            model: The model to use for generation.
            tokenizer: The tokenizer for the model.
            temperature: Sampling temperature.

        Returns:
            str: Generated text for the report section.
        """
        generation_config = GenerationConfig(
            **{
                **self.generation_args,
                "bos_token_id": model.config.bos_token_id,
                "eos_token_id": model.config.eos_token_id,
                "pad_token_id": model.config.pad_token_id,
                "decoder_start_token_id": tokenizer.cls_token_id,
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.9,
            }
        )

        generated_ids = model.generate(pixel_values, generation_config=generation_config)

        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _generate_samples_for_confidence(
        self, findings_pixels: torch.Tensor, impression_pixels: torch.Tensor,
        num_samples: int = 5, temperature: float = 0.8
    ) -> dict:
        """Generate multiple report samples for self-consistency confidence.
        
        Args:
            findings_pixels: Processed findings image tensor
            impression_pixels: Processed impression image tensor
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            
        Returns:
            Dict with samples for findings and impression
        """
        findings_samples = []
        impression_samples = []
        
        with torch.inference_mode():
            for _ in range(num_samples):
                findings_text = self._generate_report_section_sampled(
                    findings_pixels, self.findings_model, self.findings_tokenizer, temperature
                )
                impression_text = self._generate_report_section_sampled(
                    impression_pixels, self.impression_model, self.impression_tokenizer, temperature
                )
                findings_samples.append(findings_text)
                impression_samples.append(impression_text)
        
        return {
            "findings_samples": findings_samples,
            "impression_samples": impression_samples,
        }

    def _compute_report_consistency(self, samples: list) -> dict:
        """Compute self-consistency metrics for report samples.
        
        Args:
            samples: List of generated report sections
            
        Returns:
            Dict with consistency metrics
        """
        from collections import Counter
        import difflib
        
        # Normalize samples
        normalized = [s.strip().lower() for s in samples]
        
        # Exact match consistency
        counter = Counter(normalized)
        most_common, most_common_count = counter.most_common(1)[0]
        exact_consistency = most_common_count / len(samples)
        
        # Semantic similarity (using difflib for simple text similarity)
        similarities = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                ratio = difflib.SequenceMatcher(None, normalized[i], normalized[j]).ratio()
                similarities.append(ratio)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        
        # Combined consistency score
        # Weight exact consistency higher for short texts, similarity higher for long texts
        avg_length = sum(len(s) for s in normalized) / len(normalized)
        if avg_length < 100:
            consistency_score = 0.7 * exact_consistency + 0.3 * avg_similarity
        else:
            consistency_score = 0.3 * exact_consistency + 0.7 * avg_similarity
        
        # Find most representative sample (closest to centroid)
        best_idx = 0
        best_avg_sim = 0
        for i, s in enumerate(normalized):
            avg_sim = sum(
                difflib.SequenceMatcher(None, s, normalized[j]).ratio()
                for j in range(len(normalized)) if j != i
            ) / (len(normalized) - 1) if len(normalized) > 1 else 1.0
            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_idx = i
        
        return {
            "consistency_score": consistency_score,
            "exact_consistency": exact_consistency,
            "avg_similarity": avg_similarity,
            "num_unique": len(counter),
            "consensus_text": samples[best_idx],
        }

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Generate a comprehensive chest X-ray report containing both findings and impression.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager.

        Returns:
            Tuple[str, Dict]: A tuple containing the complete report and metadata.
        """
        try:
            # Process image for both models
            findings_pixels = self._process_image(
                image_path, self.findings_processor, self.findings_model
            )
            impression_pixels = self._process_image(
                image_path, self.impression_processor, self.impression_model
            )

            # Generate both sections (deterministic)
            with torch.inference_mode():
                findings_text = self._generate_report_section(
                    findings_pixels, self.findings_model, self.findings_tokenizer
                )
                impression_text = self._generate_report_section(
                    impression_pixels, self.impression_model, self.impression_tokenizer
                )

            # Generate samples for self-consistency confidence
            samples = self._generate_samples_for_confidence(
                findings_pixels, impression_pixels, num_samples=5, temperature=0.8
            )
            
            # Compute consistency scores
            findings_confidence = self._compute_report_consistency(samples["findings_samples"])
            impression_confidence = self._compute_report_consistency(samples["impression_samples"])
            
            # Overall confidence is weighted average
            overall_confidence = (
                0.6 * findings_confidence["consistency_score"] +
                0.4 * impression_confidence["consistency_score"]
            )

            # Combine into formatted report
            report = (
                "CHEST X-RAY REPORT\n\n"
                f"FINDINGS:\n{findings_text}\n\n"
                f"IMPRESSION:\n{impression_text}"
            )

            metadata = {
                "image_path": image_path,
                "analysis_status": "completed",
                "sections_generated": ["findings", "impression"],
                # Confidence-enabling data
                "confidence_data": {
                    "overall_confidence": overall_confidence,
                    "findings": {
                        "consistency_score": findings_confidence["consistency_score"],
                        "exact_consistency": findings_confidence["exact_consistency"],
                        "avg_similarity": findings_confidence["avg_similarity"],
                        "samples": samples["findings_samples"],
                    },
                    "impression": {
                        "consistency_score": impression_confidence["consistency_score"],
                        "exact_consistency": impression_confidence["exact_consistency"],
                        "avg_similarity": impression_confidence["avg_similarity"],
                        "samples": samples["impression_samples"],
                    },
                },
            }

            return report, metadata

        except Exception as e:
            return f"Error generating report: {str(e)}", {
                "image_path": image_path,
                "analysis_status": "failed",
                "error": str(e),
            }

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Asynchronously generate a comprehensive chest X-ray report."""
        return self._run(image_path)
