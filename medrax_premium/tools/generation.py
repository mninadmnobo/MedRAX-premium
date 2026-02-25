from typing import Dict, Optional, Tuple, Type
from pathlib import Path
import uuid
import tempfile
import torch
from pydantic import BaseModel, Field
from diffusers import StableDiffusionPipeline
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool


class ChestXRayGeneratorInput(BaseModel):
    """Input schema for the Chest X-Ray Generator Tool."""
    
    prompt: str = Field(
        ..., 
        description="Description of the medical condition to generate (e.g., 'big left-sided pleural effusion')"
    )
    height: int = Field(
        512,
        description="Height of generated image in pixels"
    )
    width: int = Field(
        512,
        description="Width of generated image in pixels"
    )
    num_inference_steps: int = Field(
        75,
        description="Number of denoising steps (higher = better quality but slower)"
    )
    guidance_scale: float = Field(
        4.0,
        description="How closely to follow the prompt (higher = more faithful but less diverse)"
    )


class ChestXRayGeneratorTool(BaseTool):
    """Tool for generating synthetic chest X-ray images using a fine-tuned Stable Diffusion model."""

    name: str = "chest_xray_generator"
    description: str = (
        "Generates synthetic chest X-ray images from text descriptions of medical conditions. "
        "Input: Text description of the medical finding or condition to generate, "
        "along with optional parameters for image size (height, width), "
        "quality (num_inference_steps), and prompt adherence (guidance_scale). "
        "Output: Path to the generated X-ray image and generation metadata."
    )
    args_schema: Type[BaseModel] = ChestXRayGeneratorInput

    model: StableDiffusionPipeline = None
    device: torch.device = None
    temp_dir: Path = None

    def __init__(
        self,
        model_path: str = "/model-weights/roentgen",
        cache_dir: str = "/model-weights",
        temp_dir: Optional[str] = None,
        device: Optional[str] = "cuda",
    ):
        """Initialize the chest X-ray generator tool."""
        super().__init__()
        
        self.device = torch.device(device) if device else "cuda"
        self.model = StableDiffusionPipeline.from_pretrained(
            model_path, cache_dir=cache_dir, torch_dtype=torch.float16
        )
        self.model = self.model.to(self.device)
        
        self.temp_dir = Path(temp_dir if temp_dir else tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)

    def _generate_multiple_images(
        self, prompt: str, num_images: int, num_inference_steps: int,
        guidance_scale: float, height: int, width: int
    ) -> list:
        """Generate multiple images for confidence scoring.
        
        Args:
            prompt: Text description of the medical condition
            num_images: Number of images to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            height: Height of generated image
            width: Width of generated image
            
        Returns:
            List of generated PIL images
        """
        images = []
        for _ in range(num_images):
            output = self.model(
                [prompt],
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale
            )
            images.append(output.images[0])
        return images

    def _compute_image_similarity(self, images: list) -> dict:
        """Compute similarity/consistency metrics across generated images.
        
        Args:
            images: List of generated PIL images
            
        Returns:
            Dict with consistency metrics
        """
        import numpy as np
        
        if len(images) < 2:
            return {
                "consistency_score": 1.0,
                "avg_pixel_similarity": 1.0,
                "std_similarity": 0.0,
            }
        
        # Convert images to numpy arrays
        arrays = [np.array(img.convert('L')) for img in images]  # Convert to grayscale
        
        # Compute pairwise similarities using normalized cross-correlation
        similarities = []
        for i in range(len(arrays)):
            for j in range(i + 1, len(arrays)):
                # Normalize arrays
                a1 = (arrays[i] - arrays[i].mean()) / (arrays[i].std() + 1e-8)
                a2 = (arrays[j] - arrays[j].mean()) / (arrays[j].std() + 1e-8)
                
                # Compute correlation coefficient
                corr = np.mean(a1 * a2)
                similarities.append(corr)
        
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        # Consistency score based on how similar the generated images are
        # Higher similarity = model is more confident/consistent
        consistency_score = (avg_similarity + 1) / 2  # Map from [-1,1] to [0,1]
        consistency_score = max(0, min(1, consistency_score))  # Clamp to [0,1]
        
        return {
            "consistency_score": float(consistency_score),
            "avg_pixel_similarity": float(avg_similarity),
            "std_similarity": float(std_similarity),
            "num_comparisons": len(similarities),
        }

    def _run(
        self,
        prompt: str,
        num_inference_steps: int = 75,
        guidance_scale: float = 4.0,
        height: int = 512,
        width: int = 512,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, str], Dict]:
        """Generate a chest X-ray image from a text description.

        Args:
            prompt: Text description of the medical condition to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            height: Height of generated image in pixels
            width: Width of generated image in pixels
            run_manager: Optional callback manager

        Returns:
            Tuple[Dict, Dict]: Output dictionary with image path and metadata dictionary
        """
        try:
            # Generate multiple images for consistency-based confidence
            num_samples = 3  # Generate 3 images for consistency scoring
            images = self._generate_multiple_images(
                prompt, num_samples, num_inference_steps, guidance_scale, height, width
            )
            
            # Use the first image as the primary output
            primary_image = images[0]
            
            # Compute consistency-based confidence
            confidence_data = self._compute_image_similarity(images)

            # Save primary generated image
            image_path = self.temp_dir / f"generated_xray_{uuid.uuid4().hex[:8]}.png"
            primary_image.save(image_path)
            
            # Save additional samples for reference
            sample_paths = [str(image_path)]
            for i, img in enumerate(images[1:], start=1):
                sample_path = self.temp_dir / f"generated_xray_{uuid.uuid4().hex[:8]}_sample{i}.png"
                img.save(sample_path)
                sample_paths.append(str(sample_path))

            output = {
                "image_path": str(image_path),
            }
            
            metadata = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "device": str(self.device),
                "image_size": (height, width),
                "analysis_status": "completed",
                # Confidence-enabling data
                "confidence_data": {
                    "consistency_score": confidence_data["consistency_score"],
                    "avg_pixel_similarity": confidence_data["avg_pixel_similarity"],
                    "std_similarity": confidence_data["std_similarity"],
                    "num_samples": num_samples,
                    "sample_paths": sample_paths,
                },
            }

            return output, metadata

        except Exception as e:
            return (
                {"error": str(e)},
                {
                    "prompt": prompt,
                    "analysis_status": "failed",
                    "error_details": str(e),
                }
            )

    async def _arun(
        self,
        prompt: str,
        num_inference_steps: int = 75,
        guidance_scale: float = 4.0,
        height: int = 512,
        width: int = 512,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, str], Dict]:
        """Async version of _run."""
        return self._run(prompt, num_inference_steps, guidance_scale, height, width)