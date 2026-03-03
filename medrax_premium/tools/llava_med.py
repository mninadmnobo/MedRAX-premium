from typing import Any, Dict, Optional, Tuple, Type
from pydantic import BaseModel, ConfigDict, Field

import torch

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from PIL import Image


# Lazy imports to avoid transformers compatibility issues at module load time
# These are loaded when LlavaMedTool is actually instantiated
_llava_imports_loaded = False
conv_templates = None
load_pretrained_model = None
tokenizer_image_token = None
process_images = None
IMAGE_TOKEN_INDEX = None
DEFAULT_IMAGE_TOKEN = None
DEFAULT_IM_START_TOKEN = None
DEFAULT_IM_END_TOKEN = None


def _lazy_load_llava():
    """Load LLaVA dependencies on first use."""
    global _llava_imports_loaded, conv_templates, load_pretrained_model
    global tokenizer_image_token, process_images
    global IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    
    if _llava_imports_loaded:
        return
    
    from medrax_premium.llava.conversation import conv_templates as _conv_templates
    from medrax_premium.llava.model.builder import load_pretrained_model as _load_pretrained_model
    from medrax_premium.llava.mm_utils import tokenizer_image_token as _tokenizer_image_token
    from medrax_premium.llava.mm_utils import process_images as _process_images
    from medrax_premium.llava.constants import (
        IMAGE_TOKEN_INDEX as _IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN as _DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN as _DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN as _DEFAULT_IM_END_TOKEN,
    )
    
    conv_templates = _conv_templates
    load_pretrained_model = _load_pretrained_model
    tokenizer_image_token = _tokenizer_image_token
    process_images = _process_images
    IMAGE_TOKEN_INDEX = _IMAGE_TOKEN_INDEX
    DEFAULT_IMAGE_TOKEN = _DEFAULT_IMAGE_TOKEN
    DEFAULT_IM_START_TOKEN = _DEFAULT_IM_START_TOKEN
    DEFAULT_IM_END_TOKEN = _DEFAULT_IM_END_TOKEN
    _llava_imports_loaded = True


class LlavaMedInput(BaseModel):
    """Input for the LLaVA-Med Visual QA tool. Only supports JPG or PNG images."""

    question: str = Field(..., description="The question to ask about the medical image")
    image_path: Optional[str] = Field(
        None,
        description="Path to the medical image file (optional), only supports JPG or PNG images",
    )


class LlavaMedTool(BaseTool):
    """Tool that performs medical visual question answering using LLaVA-Med.

    This tool uses a large language model fine-tuned on medical images to answer
    questions about medical images. It can handle both image-based questions and
    general medical questions without images.
    """

    name: str = "llava_med_qa"
    description: str = (
        "A tool that answers questions about biomedical images and general medical questions using LLaVA-Med. "
        "While it can process chest X-rays, it may not be as reliable for detailed chest X-ray analysis. "
        "Input should be a question and optionally a path to a medical image file."
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    args_schema: Type[BaseModel] = LlavaMedInput
    tokenizer: Any = None
    model: Any = None
    image_processor: Any = None
    context_len: int = 200000
    device: Any = None
    num_consistency_samples: int = 0
    consistency_temperature: float = 0.7

    def __init__(
        self,
        model_path: str = "microsoft/llava-med-v1.5-mistral-7b",
        cache_dir: str = "/model-weights",
        low_cpu_mem_usage: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        num_consistency_samples: int = 0,
        consistency_temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.num_consistency_samples = max(0, num_consistency_samples)
        self.consistency_temperature = consistency_temperature
        _lazy_load_llava()  # Load LLaVA dependencies on first use
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            cache_dir=cache_dir,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
            device=device,
            **kwargs,
        )
        self.model.eval()

    def _process_input(
        self, question: str, image_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.model.config.mm_use_im_start_end:
            question = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + question
            )
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
        )

        image_tensor = None
        if image_path:
            image = Image.open(image_path)
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            image_tensor = image_tensor.unsqueeze(0).half()

        return input_ids, image_tensor

    def _generate_single_response(self, input_ids: torch.Tensor, image_tensor: torch.Tensor,
                                   do_sample: bool = False, temperature: float = 0.2) -> str:
        """Generate a single response from the model.
        
        Args:
            input_ids: Tokenized input
            image_tensor: Processed image tensor
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            
        Returns:
            str: Generated response
        """
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=512,
                use_cache=True,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def _generate_samples_for_confidence(self, input_ids: torch.Tensor, image_tensor: torch.Tensor,
                                          num_samples: int = 5, temperature: float = 0.7) -> list:
        """Generate multiple samples for self-consistency confidence scoring.
        
        Args:
            input_ids: Tokenized input
            image_tensor: Processed image tensor
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            
        Returns:
            List[str]: List of generated responses
        """
        samples = []
        for _ in range(num_samples):
            response = self._generate_single_response(
                input_ids, image_tensor, do_sample=True, temperature=temperature
            )
            samples.append(response)
        return samples

    def _compute_self_consistency_score(self, samples: list) -> dict:
        """Compute self-consistency confidence score from multiple samples.
        
        Args:
            samples: List of generated responses
            
        Returns:
            Dict with confidence metrics
        """
        from collections import Counter
        
        # Normalize samples for comparison
        normalized = [s.strip().lower() for s in samples]
        counter = Counter(normalized)
        most_common_answer, most_common_count = counter.most_common(1)[0]
        
        # Self-consistency score
        consistency_score = most_common_count / len(samples)
        
        # Find original most common answer
        most_common_original = samples[0]
        for s in samples:
            if s.strip().lower() == most_common_answer:
                most_common_original = s
                break
        
        return {
            "consistency_score": consistency_score,
            "num_samples": len(samples),
            "num_unique_answers": len(counter),
            "most_common_count": most_common_count,
            "answer_distribution": dict(counter),
            "consensus_answer": most_common_original,
        }

    def _run(
        self,
        question: str,
        image_path: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Answer a medical question, optionally based on an input image.

        Args:
            question (str): The medical question to answer.
            image_path (Optional[str]): The path to the medical image file (if applicable).
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.

        Returns:
            Tuple[str, Dict]: A tuple containing the model's answer and any additional metadata.

        Raises:
            Exception: If there's an error processing the input or generating the answer.
        """
        try:
            input_ids, image_tensor = self._process_input(question, image_path)
            input_ids = input_ids.to(device=self.model.device)
            if image_tensor is not None:
                image_tensor = image_tensor.to(device=self.model.device, dtype=self.model.dtype)

            # Generate primary response (deterministic)
            output = self._generate_single_response(input_ids, image_tensor, do_sample=False)

            # Build metadata
            metadata = {
                "question": question,
                "image_path": image_path,
                "analysis_status": "completed",
            }

            # Optional self-consistency confidence scoring
            if self.num_consistency_samples > 0:
                samples = self._generate_samples_for_confidence(
                    input_ids, image_tensor,
                    num_samples=self.num_consistency_samples,
                    temperature=self.consistency_temperature,
                )
                confidence_data = self._compute_self_consistency_score(samples)
                metadata["confidence_data"] = {
                    "samples": samples,
                    "consistency_score": confidence_data["consistency_score"],
                    "num_unique_answers": confidence_data["num_unique_answers"],
                    "answer_distribution": confidence_data["answer_distribution"],
                    "consensus_answer": confidence_data["consensus_answer"],
                }

            return output, metadata
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__}: {e!r}"
            print(f"  ❌ LLaVA-Med error: {err_msg}")
            return {"error": err_msg}, {
                "question": question,
                "image_path": image_path,
                "analysis_status": "failed",
                "error_details": err_msg,
            }

    async def _arun(
        self,
        question: str,
        image_path: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, Dict]:
        """Asynchronously answer a medical question, optionally based on an input image.

        This method currently calls the synchronous version, as the model inference
        is not inherently asynchronous. For true asynchronous behavior, consider
        using a separate thread or process.

        Args:
            question (str): The medical question to answer.
            image_path (Optional[str]): The path to the medical image file (if applicable).
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.

        Returns:
            Tuple[str, Dict]: A tuple containing the model's answer and any additional metadata.

        Raises:
            Exception: If there's an error processing the input or generating the answer.
        """
        return self._run(question, image_path)
