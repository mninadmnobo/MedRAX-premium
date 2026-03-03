from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import tempfile
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class XRayPhraseGroundingInput(BaseModel):
    """Input schema for the XRay Phrase Grounding Tool. Only supports JPG or PNG images."""

    image_path: str = Field(
        ...,
        description="Path to the frontal chest X-ray image file, only supports JPG or PNG images",
    )
    phrase: str = Field(
        ...,
        description="Medical finding or condition to locate in the image (e.g., 'Pleural effusion')",
    )
    max_new_tokens: int = Field(default=256, description="Maximum number of new tokens to generate")


class XRayPhraseGroundingTool(BaseTool):
    """Tool for grounding medical findings in chest X-ray images using the MAIRA-2 model.

    This tool processes chest X-ray images and locates specific medical findings mentioned
    in the input phrase. It returns both the bounding box coordinates and a visualization
    of the finding's location in the image.
    """

    name: str = "xray_phrase_grounding"
    description: str = (
        "Locates and visualizes specific medical findings in chest X-ray images. "
        "Takes a chest X-ray image and medical phrase to locate (e.g., 'Pleural effusion', 'Cardiomegaly'). "
        "Returns bounding box coordinates in format [x_topleft, y_topleft, x_bottomright, y_bottomright] "
        "where each value is between 0-1 representing relative position in the image, "
        "a visualization of the finding's location, and confidence metadata. "
        "Example input: {'image_path': '/path/to/xray.png', 'phrase': 'Pleural effusion', 'max_new_tokens': 300}"
    )
    args_schema: Type[BaseModel] = XRayPhraseGroundingInput

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = None
    processor: Any = None
    device: Any = None
    temp_dir: Path = None

    def __init__(
        self,
        model_path: str = "microsoft/maira-2",
        cache_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device: Optional[str] = "cuda",
    ):
        """Initialize the XRay Phrase Grounding Tool."""
        import gc
        super().__init__()
        self.device = torch.device(device) if device else "cuda"

        # -- Pre-flight: GPU free-space guard --------------------------------
        quantized = load_in_4bit or load_in_8bit
        from medrax_premium.tools.xray_vqa import _check_gpu_free_space, _parse_gpu_id
        _check_gpu_free_space(
            str(self.device),
            required_gb=4.0 if quantized else 9.0,
            label="MAIRA-2 Grounding",
        )

        # Setup quantization config
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        # Fix: MAIRA-2 has no `lm_head` attribute, which crashes
        # bitsandbytes quantization (get_keys_to_not_convert → get_output_embeddings
        # → self.lm_head → AttributeError). Patch before loading.
        _bnb_int = None
        _orig_get_keys = None
        if quantization_config is not None:
            import transformers.integrations.bitsandbytes as _bnb_int
            if hasattr(_bnb_int, "get_keys_to_not_convert"):
                _orig_get_keys = _bnb_int.get_keys_to_not_convert
                def _safe_get_keys(model):
                    try:
                        return _orig_get_keys(model)
                    except AttributeError:
                        return []
                _bnb_int.get_keys_to_not_convert = _safe_get_keys

        # -- Device-map: pin quantised model to *target* GPU only -----------
        device_kw: Dict = {}
        if quantization_config is not None:
            gpu_id = _parse_gpu_id(str(self.device))
            max_mem: Dict = {}
            for i in range(torch.cuda.device_count()):
                if i == gpu_id:
                    tot = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    max_mem[i] = f"{int(tot) - 1}GiB"
                else:
                    max_mem[i] = "0GiB"
            max_mem["cpu"] = "24GiB"
            device_kw["device_map"] = "auto"
            device_kw["max_memory"] = max_mem
        else:
            device_kw["device_map"] = {"": str(self.device)}

        gc.collect()
        torch.cuda.empty_cache()

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            quantization_config=quantization_config,
            **device_kw,
        )

        # Restore original function after loading
        if _bnb_int is not None and _orig_get_keys is not None:
            _bnb_int.get_keys_to_not_convert = _orig_get_keys
        self.processor = AutoProcessor.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=True
        )

        
        self.model = self.model.eval()

        self.temp_dir = Path(temp_dir if temp_dir else tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)

    def _visualize_bboxes(
        self, image: Image.Image, bboxes: List[Tuple[float, float, float, float]], phrase: str
    ) -> str:
        """Create and save visualization of multiple bounding boxes on the image."""
        plt.figure(figsize=(12, 12))
        plt.imshow(image, cmap="gray")

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            plt.gca().add_patch(
                plt.Rectangle(
                    (x1 * image.width, y1 * image.height),
                    width * image.width,
                    height * image.height,
                    fill=False,
                    color="red",                    linewidth=2,
                )
            )

        plt.title(f"Located: {phrase}", pad=20)
        plt.axis("off")

        viz_path = self.temp_dir / f"grounding_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(viz_path, bbox_inches="tight", dpi=150)
        plt.close()

        return str(viz_path)

    def _compute_grounding_confidence(self, predictions: list, image_size: tuple) -> dict:
        """Compute confidence metrics for grounding predictions.
        
        Args:
            predictions: List of processed predictions with bounding boxes
            image_size: (width, height) of the image
            
        Returns:
            Dict with confidence metrics
        """
        if not predictions:
            return {
                "has_prediction": False,
                "confidence_score": 0.0,
                "num_boxes": 0,
                "coverage_ratio": 0.0,
                "box_metrics": [],
            }
        
        img_width, img_height = image_size
        img_area = img_width * img_height
        
        box_metrics = []
        total_coverage = 0.0
        
        for pred in predictions:
            for bbox in pred["bounding_boxes"]["image_coordinates"]:
                x1, y1, x2, y2 = bbox
                # Box dimensions (normalized 0-1)
                box_width = (x2 - x1) * img_width
                box_height = (y2 - y1) * img_height
                box_area = box_width * box_height
                
                # Coverage ratio
                coverage = box_area / img_area if img_area > 0 else 0
                total_coverage += coverage
                
                # Aspect ratio (closer to 1 is often more confident for medical findings)
                aspect_ratio = box_width / box_height if box_height > 0 else 0
                
                # Center position (findings near center may be more reliable)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                center_dist = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
                
                box_metrics.append({
                    "box_area_ratio": coverage,
                    "aspect_ratio": aspect_ratio,
                    "center_distance": center_dist,
                    "bbox": bbox,
                })
        
        num_boxes = len(box_metrics)
        
        # Confidence heuristics:
        # - Having predictions is a positive signal
        # - Reasonable coverage (not too small, not too large)
        # - Multiple consistent boxes may indicate higher confidence
        
        # Base confidence from having predictions
        base_confidence = 0.5 if num_boxes > 0 else 0.0
        
        # Coverage bonus (optimal coverage is 0.01-0.3 of image)
        avg_coverage = total_coverage / num_boxes if num_boxes > 0 else 0
        if 0.01 <= avg_coverage <= 0.3:
            coverage_bonus = 0.3
        elif avg_coverage < 0.01:
            coverage_bonus = avg_coverage * 30  # Scale up small coverages
        else:
            coverage_bonus = max(0, 0.3 - (avg_coverage - 0.3))  # Penalize very large boxes
        
        # Consistency bonus for multiple boxes
        consistency_bonus = min(0.2, (num_boxes - 1) * 0.05) if num_boxes > 1 else 0
        
        confidence_score = min(1.0, base_confidence + coverage_bonus + consistency_bonus)
        
        return {
            "has_prediction": True,
            "confidence_score": confidence_score,
            "num_boxes": num_boxes,
            "coverage_ratio": total_coverage,
            "avg_coverage": avg_coverage,
            "box_metrics": box_metrics,
        }

    def _run(
        self,
        image_path: str,
        phrase: str,
        max_new_tokens: int = 300,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Ground a medical finding phrase in an X-ray image.

        Args:
            image_path: Path to the chest X-ray image file
            phrase: Medical finding to locate in the image
            max_new_tokens: Maximum number of new tokens to generate
            run_manager: Optional callback manager

        Returns:
            Tuple[Dict, Dict]: Output dictionary and metadata dictionary
        """
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor.format_and_preprocess_phrase_grounding_input(
                frontal_image=image, phrase=phrase, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )

            prompt_length = inputs["input_ids"].shape[-1]
            decoded_text = self.processor.decode(
                output[0][prompt_length:], skip_special_tokens=True
            )
            predictions = self.processor.convert_output_to_plaintext_or_grounded_sequence(
                decoded_text
            )

            metadata = {
                "image_path": image_path,
                "original_size": image.size,
                "model_input_size": tuple(inputs["pixel_values"].shape[-2:]),
                "device": str(self.device),
                "analysis_status": "completed",
            }

            if not predictions:
                confidence_data = self._compute_grounding_confidence([], image.size)
                output = {
                    "predictions": [],
                    "visualization_path": None,
                }
                metadata["analysis_status"] = "completed_no_finding"
                metadata["confidence_data"] = confidence_data
                return output, metadata

            # Process multiple predictions
            processed_predictions = []
            for pred_phrase, pred_bboxes in predictions:
                if not pred_bboxes:  # Skip if no bounding boxes
                    continue

                # Convert model bboxes to list format and get original image bboxes
                model_bboxes = [list(bbox) for bbox in pred_bboxes]
                original_bboxes = [
                    self.processor.adjust_box_for_original_image_size(
                        bbox, width=image.size[0], height=image.size[1]
                    )
                    for bbox in model_bboxes
                ]

                processed_predictions.append(
                    {
                        "phrase": pred_phrase,
                        "bounding_boxes": {
                            "model_coordinates": model_bboxes,
                            "image_coordinates": original_bboxes,
                        },
                    }
                )

            # Compute grounding confidence
            confidence_data = self._compute_grounding_confidence(processed_predictions, image.size)

            # Create visualization with all bounding boxes
            if processed_predictions:
                all_bboxes = []
                for pred in processed_predictions:
                    all_bboxes.extend(pred["bounding_boxes"]["image_coordinates"])
                viz_path = self._visualize_bboxes(image, all_bboxes, phrase)
            else:
                viz_path = None
                metadata["analysis_status"] = "completed_no_finding"

            output = {
                "predictions": processed_predictions,
                "visualization_path": viz_path,
            }
            
            # Add confidence data to metadata
            metadata["confidence_data"] = confidence_data

            return output, metadata

        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(
        self,
        image_path: str,
        phrase: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Asynchronous version of _run."""
        return self._run(image_path, phrase, run_manager)
