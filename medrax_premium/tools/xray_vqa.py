"""
XRayVQATool — CheXagent-based chest X-ray visual question answering.

Premium enhancements over base MedRAX
--------------------------------------
* Robust input parsing (``image_paths`` accepts ``str``, ``List[str]``, or
  JSON-encoded list) so the LLM agent can pass any shape.
* Transformers version faking at **both** load-time *and* inference-time to
  satisfy CheXagent's ``trust_remote_code`` assertions on any transformers ≥4.40.
* Configurable self-consistency confidence scoring (``num_consistency_samples``)
  that feeds ``VQAConfidenceExtractor`` → ``CanonicalFinding`` pipeline.
  Set to 0 for fast benchmark runs; the canonical-output layer still works
  via keyword-analysis fallback.
* Detailed, type-prefixed error messages for every failure path so the agent
  can surface them.
* ``model_config`` with ``arbitrary_types_allowed`` for Pydantic v2.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Any, Union
from pathlib import Path
from collections import Counter
from pydantic import BaseModel, ConfigDict, Field, field_validator
import json

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# GPU helpers (shared by all quantised tools)
# ---------------------------------------------------------------------------

def _parse_gpu_id(device) -> int:
    """Extract integer GPU index from a device string like 'cuda:1'."""
    s = str(device)
    if ":" in s:
        return int(s.split(":")[-1])
    return 0


def _check_gpu_free_space(
    device, required_gb: float = 5.0, label: str = "model",
) -> None:
    """Raise early if the target GPU lacks enough free VRAM."""
    if not torch.cuda.is_available():
        return
    gpu_id = _parse_gpu_id(device)
    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"[{label}] GPU-{gpu_id} does not exist "
            f"(only {torch.cuda.device_count()} GPU(s) visible)"
        )
    free_gb = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3
    total_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
    print(
        f"  [{label}] GPU-{gpu_id} pre-load: "
        f"{free_gb:.2f}/{total_gb:.2f} GB free  "
        f"(need ~{required_gb:.1f} GB)"
    )
    if free_gb < required_gb:
        raise RuntimeError(
            f"[{label}] GPU-{gpu_id} only {free_gb:.2f} GB free, "
            f"need ~{required_gb:.1f} GB. Free VRAM or use 4-bit."
        )


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class XRayVQAToolInput(BaseModel):
    """Input schema for the CheXagent Tool.

    ``image_paths`` is intentionally typed as ``Union[List[str], str]`` so the
    LLM agent can pass a bare string, a JSON-encoded list, or a Python list.
    The ``field_validator`` normalises all variants into ``List[str]``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_paths: Union[List[str], str] = Field(
        ..., description="List of paths to chest X-ray images to analyze"
    )
    prompt: str = Field(
        ..., description="Question or instruction about the chest X-ray images"
    )
    max_new_tokens: int = Field(
        512, description="Maximum number of tokens to generate in the response"
    )

    @field_validator("image_paths", mode="before")
    @classmethod
    def parse_image_paths(cls, v: Any) -> List[str]:
        """Normalise ``image_paths`` from any input shape into ``List[str]``."""
        if isinstance(v, str):
            v = v.strip()
            # "['/a.png', '/b.png']" → try JSON first
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(p) for p in parsed]
                except json.JSONDecodeError:
                    v = v[1:-1].strip()
            # Comma-separated fallback
            if "," in v:
                paths = [p.strip().strip('"').strip("'") for p in v.split(",")]
                return [p for p in paths if p]
            # Single path
            return [v]
        if isinstance(v, (list, tuple)):
            return [str(p) for p in v]
        return v


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class XRayVQATool(BaseTool):
    """CheXagent-powered chest X-ray visual question answering.

    Produces ``(output_dict, metadata_dict)`` tuples whose ``metadata``
    contains a ``confidence_data`` block consumed by
    ``VQAConfidenceExtractor`` → ``normalize_vqa_output`` →
    ``CanonicalFinding``  in the premium conflict-resolution pipeline.
    """

    # -- Pydantic v2 config (torch.dtype is not JSON-serialisable) ----------
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "chest_xray_expert"
    description: str = (
        "A versatile tool for analyzing chest X-rays. "
        "Can perform multiple tasks including: visual question answering, "
        "report generation, abnormality detection, comparative analysis, "
        "anatomical description, and clinical interpretation. "
        "Input should be paths to X-ray images and a natural language prompt "
        "describing the analysis needed."
    )
    args_schema: Type[BaseModel] = XRayVQAToolInput
    return_direct: bool = True

    # -- Fields (set in __init__) -------------------------------------------
    cache_dir: Optional[str] = None
    device: Optional[str] = None
    dtype: torch.dtype = torch.bfloat16
    tokenizer: Optional[Any] = None          # AutoTokenizer
    model: Optional[Any] = None              # AutoModelForCausalLM
    _processor: Optional[Any] = None         # AutoProcessor (BLIP-2 only)
    _is_blip2: bool = False                  # True for CheXagent-8b
    num_consistency_samples: int = 0         # 0 = fast mode (no extra passes)
    consistency_temperature: float = 0.7

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------
    def __init__(
        self,
        model_name: str = "StanfordAIMI/CheXagent-2-3b",
        device: Optional[str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        num_consistency_samples: int = 0,
        consistency_temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        """Initialise the XRayVQATool.

        Args:
            model_name: HuggingFace repo ID **or** path to a local directory
                containing the CheXagent weights (e.g. a Kaggle dataset mount).
            device: ``"cuda"``, ``"cuda:0"``, ``"cpu"``, etc.
            dtype: Weight dtype (default ``bfloat16``).
            cache_dir: Directory for downloaded model snapshots.
            load_in_4bit: Use NF4 quantisation (~2 GB VRAM).
            load_in_8bit: Use 8-bit quantisation (~4 GB VRAM).
            num_consistency_samples: Number of extra stochastic samples for
                self-consistency confidence scoring.  **0** disables it
                (recommended for large benchmarks); the canonical-output
                layer will fall back to keyword-analysis confidence.
            consistency_temperature: Sampling temperature for self-consistency.
            **kwargs: Forwarded to ``BaseTool.__init__``.
        """
        super().__init__(**kwargs)

        import gc
        from transformers import BitsAndBytesConfig

        # Fake version for load-time remote-code assertions
        original_ver = transformers.__version__
        transformers.__version__ = "4.40.0"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.num_consistency_samples = max(0, num_consistency_samples)
        self.consistency_temperature = consistency_temperature

        # -- Pre-flight: GPU free-space guard --------------------------------
        quantized = load_in_4bit or load_in_8bit
        _check_gpu_free_space(
            self.device, required_gb=5.0 if quantized else 10.0,
            label="CheXagent",
        )

        # Download / locate weights and patch version asserts in remote code
        local_path = self._download_and_patch_model(model_name, cache_dir)

        # Quantisation config
        quant_kwargs: Dict[str, Any] = {}
        if load_in_4bit:
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # -- Device-map: pin quantised model to *target* GPU only -----------
        # device_map="auto" with no max_memory can spill onto the other GPU,
        # breaking the balanced 2-GPU layout.  We constrain accelerate to
        # the single intended GPU via max_memory.
        device_kwargs: Dict[str, Any] = {}
        if quant_kwargs:
            gpu_id = _parse_gpu_id(self.device)
            max_mem: Dict = {}
            for i in range(torch.cuda.device_count()):
                if i == gpu_id:
                    # Use ~55% of total to leave headroom for quantisation
                    # staging.  Layers that don't fit are CPU-offloaded and
                    # moved to GPU after quantisation.
                    tot = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    free = torch.cuda.mem_get_info(i)[0] / 1024**3
                    budget = min(free - 1.0, tot * 0.55)
                    max_mem[i] = f"{max(4, int(budget))}GiB"
                else:
                    max_mem[i] = "0GiB"          # block other GPUs
            max_mem["cpu"] = "32GiB"             # CPU offload safety-net
            device_kwargs["device_map"] = "auto"
            device_kwargs["max_memory"] = max_mem
            print(f"  [CheXagent] max_memory = {max_mem}")
        else:
            device_kwargs["device_map"] = {"": str(self.device)}

        # Free fragmented VRAM before the heavy load
        gc.collect()
        torch.cuda.empty_cache()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True, cache_dir=cache_dir,
        )
        # CheXagent tokenizers cached locally sometimes lack a chat_template.
        # Inject a QWen-compatible Jinja2 template (from/value key format).
        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['from'] == 'system' %}<|im_start|>system\n{{ message['value'] }}<|im_end|>\n"
                "{% elif message['from'] == 'human' %}<|im_start|>user\n{{ message['value'] }}<|im_end|>\n"
                "{% elif message['from'] == 'gpt' %}<|im_start|>assistant\n{{ message['value'] }}<|im_end|>\n"
                "{% endif %}{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            )
            print("  [CheXagent] chat_template missing — applied QWen fallback")

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
            **device_kwargs,
            **quant_kwargs,
        )
        if not quantized:
            self.model = self.model.to(dtype=self.dtype)
        self.model.eval()

        # Detect architecture: BLIP-2 (CheXagent-8b) vs Qwen-VL (CheXagent-2-3b)
        # BLIP-2 has a vision_model + qformer and its generate() expects pixel_values.
        self._is_blip2 = hasattr(self.model, 'vision_model')
        if self._is_blip2:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                local_path, trust_remote_code=True, cache_dir=cache_dir,
            )
            print("  [CheXagent] BLIP-2 architecture detected — using AutoProcessor")

        transformers.__version__ = original_ver

    # -----------------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------------
    @staticmethod
    def _download_and_patch_model(
        model_name: str, cache_dir: Optional[str] = None
    ) -> str:
        """Download model and patch version-check asserts.

        Accepts either a HuggingFace repo ID (downloaded via
        ``snapshot_download``) or a local directory path (e.g. a Kaggle
        dataset mount), which is used directly.
        """
        import glob, os

        if os.path.isdir(model_name):
            local_path = model_name
        else:
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(model_name, cache_dir=cache_dir)

        for py_file in glob.glob(str(Path(local_path) / "*.py")):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                if 'assert transformers.__version__' in content:
                    content = content.replace(
                        'assert transformers.__version__ == "4.40.0"',
                        '# assert transformers.__version__ == "4.40.0"  # Patched',
                    )
                    with open(py_file, "w", encoding="utf-8") as f:
                        f.write(content)
            except Exception:
                pass

        return local_path

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------
    def _generate_response(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        """Generate a response with transformers version faked at runtime."""
        saved = transformers.__version__
        transformers.__version__ = "4.40.0"
        try:
            return self._generate_response_inner(
                image_paths, prompt, max_new_tokens,
                do_sample, temperature, top_p,
            )
        finally:
            transformers.__version__ = saved

    def _generate_response_inner(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        """Core generation — handles both CheXagent-8b (BLIP-2) and
        CheXagent-2-3b (Qwen-VL) architectures automatically.
        """
        _model_device = next(self.model.parameters()).device

        if self._is_blip2:
            # ---- BLIP-2 / CheXagent-8b path ----
            # generate() expects pixel_values (5-D) + input_ids as kwargs.
            # AutoProcessor handles PIL loading + tokenization together.
            from PIL import Image as _PILImage
            pil_images = [_PILImage.open(p).convert("RGB") for p in image_paths]
            text = f" USER: <s>{prompt} ASSISTANT: <s>"
            inputs = self._processor(
                images=pil_images, text=text, return_tensors="pt",
            )
            # Move each tensor to model device; float tensors get model dtype
            inputs = {
                k: v.to(device=_model_device, dtype=self.dtype)
                   if v.is_floating_point()
                   else v.to(device=_model_device)
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    num_beams=1,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=True,
                    max_new_tokens=max_new_tokens,
                )[0]
                return self._processor.tokenizer.decode(
                    output, skip_special_tokens=True,
                )

        # ---- Qwen-VL / CheXagent-2-3b path (original MedRAX) ----
        list_format = [*[{"image": p} for p in image_paths], {"text": prompt}]
        try:
            query = self.tokenizer.from_list_format(list_format)
        except (AttributeError, Exception):
            query = ""
            num_images = 0
            for ele in list_format:
                if "image" in ele:
                    num_images += 1
                    query += f'Picture {num_images}: <img>{ele["image"]}</img>\n'
                elif "text" in ele:
                    query += ele["text"]

        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        tokenized = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )
        if isinstance(tokenized, torch.Tensor):
            input_ids = tokenized
        else:
            _raw = tokenized["input_ids"]
            input_ids = (
                _raw if isinstance(_raw, torch.Tensor)
                else torch.tensor(_raw, dtype=torch.long)
            )
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device=_model_device)

        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                do_sample=do_sample,
                num_beams=1,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                max_new_tokens=max_new_tokens,
            )[0]
            return self.tokenizer.decode(output[input_ids.size(1):-1])

    # -----------------------------------------------------------------------
    # Self-consistency confidence (optional — controlled by __init__ param)
    # -----------------------------------------------------------------------
    def _generate_samples_for_confidence(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int,
        num_samples: int = 5,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate *num_samples* stochastic answers for self-consistency."""
        samples: List[str] = []
        for _ in range(num_samples):
            resp = self._generate_response(
                image_paths, prompt, max_new_tokens,
                do_sample=True, temperature=temperature, top_p=0.9,
            )
            samples.append(resp)
        return samples

    @staticmethod
    def _compute_self_consistency_score(samples: List[str]) -> Dict[str, Any]:
        """Return self-consistency metrics consumed by ``VQAConfidenceExtractor``.

        Keys produced match the contract in
        ``medrax_premium.agent.confidence_scoring.VQAConfidenceExtractor``:
        ``consistency_score``, ``samples``, ``num_unique_answers``,
        ``answer_distribution``, ``consensus_answer``.
        """
        normalised = [s.strip().lower() for s in samples]
        counter = Counter(normalised)
        best_answer, best_count = counter.most_common(1)[0]
        consistency = best_count / len(samples)

        # Map back to original casing
        consensus = next(s for s in samples if s.strip().lower() == best_answer)

        return {
            "consistency_score": consistency,
            "num_samples": len(samples),
            "num_unique_answers": len(counter),
            "most_common_count": best_count,
            "answer_distribution": dict(counter),
            "consensus_answer": consensus,
        }

    # -----------------------------------------------------------------------
    # _run  (synchronous entry-point)
    # -----------------------------------------------------------------------
    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute chest X-ray analysis.

        Returns:
            ``(output_dict, metadata_dict)`` — the metadata always contains a
            ``confidence_data`` block (possibly empty if self-consistency is
            disabled) so that ``normalize_vqa_output`` and the
            ``VQAConfidenceExtractor`` can score the finding.
        """
        try:
            # Verify every image exists
            for p in image_paths:
                if not Path(p).is_file():
                    raise FileNotFoundError(f"Image file not found: {p}")

            # Primary deterministic response
            response = self._generate_response(
                image_paths, prompt, max_new_tokens,
            )

            # Optional self-consistency confidence
            confidence_data: Dict[str, Any] = {}
            if self.num_consistency_samples > 0:
                samples = self._generate_samples_for_confidence(
                    image_paths, prompt, max_new_tokens,
                    num_samples=self.num_consistency_samples,
                    temperature=self.consistency_temperature,
                )
                sc = self._compute_self_consistency_score(samples)
                confidence_data = {
                    "samples": samples,
                    "consistency_score": sc["consistency_score"],
                    "num_unique_answers": sc["num_unique_answers"],
                    "answer_distribution": sc["answer_distribution"],
                    "consensus_answer": sc["consensus_answer"],
                }

            output = {"response": response}
            metadata: Dict[str, Any] = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "completed",
            }
            if confidence_data:
                metadata["confidence_data"] = confidence_data

            return output, metadata

        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            err_msg = (
                f"{type(e).__name__}: {e}" if str(e)
                else f"{type(e).__name__}: {e!r}"
            )
            print(f"  \u274c CheXagent _run error: {err_msg}")
            return (
                {"error": err_msg},
                {
                    "image_paths": image_paths,
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                    "analysis_status": "failed",
                    "error_details": err_msg,
                },
            )

    # -----------------------------------------------------------------------
    # _arun  (async entry-point — delegates to sync)
    # -----------------------------------------------------------------------
    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Async version — delegates to ``_run``."""
        return self._run(image_paths, prompt, max_new_tokens)
