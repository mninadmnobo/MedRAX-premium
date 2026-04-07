import os
import warnings
from pathlib import Path
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool

from interface import create_demo
from medrax_premium.agent import *
from medrax_premium.tools import *
from medrax_premium.tools.grounding import XRayPhraseGroundingTool
from medrax_premium.utils import *

try:
    from medrax_premium.tools.generation import ChestXRayGeneratorTool
except Exception:
    ChestXRayGeneratorTool = None

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


def _resolve_work_path(*parts):
    """Return a writable path under the active work directory."""
    work_root = Path(os.environ.get("WORK_DIR", "/kaggle/working"))
    return work_root.joinpath(*parts)


def _resolve_model(model_dir, local_subpath, hf_repo_id):
    """Return a local path if model_dir/local_subpath exists, else the HF repo ID.

    This lets the same code work with:
      - Flat local directories (e.g. Kaggle datasets)
      - HF cache directories (original Docker / local dev setup)
    """
    target_parts = Path(local_subpath).parts

    direct = Path(model_dir).joinpath(*target_parts)
    if direct.is_dir():
        return str(direct)

    nested = direct.joinpath(*target_parts)
    if nested.is_dir():
        return str(nested)

    search_roots = [Path(model_dir), Path("/kaggle/input"), Path("/kaggle/working")]
    seen_roots = set()
    for root in search_roots:
        if not root.is_dir() or root in seen_roots:
            continue
        seen_roots.add(root)
        try:
            for candidate in root.rglob(target_parts[-1]):
                if candidate.is_dir() and tuple(candidate.parts[-len(target_parts):]) == target_parts:
                    return str(candidate)
        except OSError:
            pass

    return hf_repo_id


def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/model-weights",
    temp_dir=None,
    device="cuda",
    model="chatgpt-4o-latest",
    temperature=0.7,
    top_p=0.95,
    openai_kwargs={}
):
    """Initialize the MedRAX agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "chatgpt-4o-latest".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        openai_kwargs (dict, optional): Additional keyword arguments for OpenAI API, such as API key and base URL.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    # Resolve model paths — prefer local flat dirs, fall back to HF repo IDs
    vqa_id = _resolve_model(model_dir, "models-chexagent", "StanfordAIMI/CheXagent-2-3b")
    llava_id = _resolve_model(model_dir, "models-llava-med", "microsoft/llava-med-v1.5-mistral-7b")
    maira_id = _resolve_model(model_dir, "models-medical-tools/maira-2", "microsoft/maira-2")
    findings_id = _resolve_model(model_dir, "models-core/swinv2_findings", "IAMJB/chexpert-mimic-cxr-findings-baseline")
    impression_id = _resolve_model(model_dir, "models-core/swinv2_impression", "IAMJB/chexpert-mimic-cxr-impression-baseline")

    if temp_dir is None:
        temp_dir = _resolve_work_path("temp")
    else:
        temp_dir = Path(temp_dir)
        if not temp_dir.is_absolute():
            temp_dir = _resolve_work_path(str(temp_dir))

    log_dir = _resolve_work_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # --- GPU memory manager: only one heavy tool on GPU 1 at a time --------
    gpu1_mgr = GPUMemoryManager()

    # Tools that are small / CPU-only — loaded eagerly on GPU 0
    eager_tools = {
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device, temp_dir=temp_dir),
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(
            cache_dir=model_dir, device=device,
            findings_model_path=findings_id,
            impression_model_path=impression_id,
        ),
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),
    }

    # Heavy tools — wrapped with LazyTool so they load on first call and
    # GPUMemoryManager swaps them out to keep GPU 1 within 16 GB.
    lazy_tools = {
        "XRayVQATool": LazyTool(
            XRayVQATool, gpu1_mgr,
            model_name=vqa_id, cache_dir=model_dir, device=device, load_in_8bit=True,
        ),
        "LlavaMedTool": LazyTool(
            LlavaMedTool, gpu1_mgr,
            model_path=llava_id, cache_dir=model_dir, device=device, load_in_8bit=True,
        ),
        "XRayPhraseGroundingTool": LazyTool(
            XRayPhraseGroundingTool, gpu1_mgr,
            model_path=maira_id, cache_dir=model_dir, temp_dir=temp_dir, load_in_4bit=True, device=device,
        ),
    }

    if ChestXRayGeneratorTool is not None:
        lazy_tools["ChestXRayGeneratorTool"] = LazyTool(
            ChestXRayGeneratorTool, gpu1_mgr,
            model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device,
        )

    all_tools = {**eager_tools, **lazy_tools}

    # Initialize only selected tools or all if none specified
    tools_dict = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            t = all_tools[tool_name]
            # Eager tools are lambdas → call them; LazyTool instances are ready as-is
            tools_dict[tool_name] = t() if callable(t) and not isinstance(t, BaseTool) else t

    checkpointer = MemorySaver()
    model = ChatOpenAI(model=model, temperature=temperature, top_p=top_p, **openai_kwargs)
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir=str(log_dir),
        system_prompt=prompt,
        checkpointer=checkpointer,
        enable_conflict_resolution=True,  # NEW: Enable conflict resolution framework
        conflict_sensitivity=0.4,  # Detect conflicts with 40% confidence gap
        deferral_threshold=0.6,  # Defer to human if confidence < 60%
    )

    print("Agent initialized with conflict resolution enabled")
    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the MedRAX application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    # Example: initialize with only specific tools
    # Here three tools are commented out, you can uncomment them to use them
    selected_tools = [
        "ImageVisualizerTool",
        "DicomProcessorTool",
        "ChestXRayClassifierTool",
        "ChestXRaySegmentationTool",
        "ChestXRayReportGeneratorTool",
        "XRayVQATool",
        "LlavaMedTool",
        # "XRayPhraseGroundingTool",
        # "ChestXRayGeneratorTool",
    ]    # Collect the ENV variables
    openai_kwargs = {}
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()

    if api_key:
        openai_kwargs["api_key"] = api_key

    # GitHub PATs must target the GitHub Models inference endpoint.
    if api_key.startswith(("github_", "ghp_", "github_pat_")) and not base_url:
        base_url = "https://models.inference.ai.azure.com"

    if base_url:
        openai_kwargs["base_url"] = base_url

    agent, tools_dict = initialize_agent(
        "medrax_premium/docs/system_prompts.txt",
        tools_to_use=selected_tools,
        model_dir=os.environ.get("MODEL_DIR", "/model-weights"),
        temp_dir=_resolve_work_path("temp"),
        device=os.environ.get("DEVICE", "cuda"),
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        temperature=0.7,
        top_p=0.95,
        openai_kwargs=openai_kwargs
    )
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name="0.0.0.0", server_port=8585, share=True)
