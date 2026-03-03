# MedRAX Premium — Cell-by-Cell Pipeline Analysis


---

## GPU Layout (Balanced ~5.5 / 5 GB on 2×T4)

```
GPU-0 (cuda:0)                              GPU-1 (cuda:1)
┌────────────────────────────────────┐      ┌────────────────────────────────┐
│ Report (SwinV2×2, fp16)     ~0.5 GB│      │ CheXagent 8B (NF4 4-bit)      │
│ Classification (DenseNet)   ~0.03  │      │                      ~5.0 GB  │
│ Segmentation (PSPNet)       ~0.1   │      └────────────────────────────────┘
│ LLaVA-Med 7B (NF4 4-bit)   ~4.5   │
│ ───────────────────────────────────│
│ Total                       ≈5.5 GB│
└────────────────────────────────────┘
```

| Tool | Class | GPU | Precision | ~VRAM |
|------|-------|:---:|-----------|------:|
| Report Generator | `ChestXRayReportGeneratorTool` | 0 | fp16 | 0.5 GB |
| Classifier | `ChestXRayClassifierTool` | 0 | fp32 | 0.03 GB |
| Segmentation | `ChestXRaySegmentationTool` | 0 | fp32 | 0.1 GB |
| LLaVA-Med 7B | `LlavaMedTool` | 0 | **NF4 4-bit** | 4.5 GB |
| CheXagent 8B | `XRayVQATool` | 1 | **NF4 4-bit** | 5.0 GB |

**Device-pinning strategy** (both 4-bit models):
`device_map="auto"` + `max_memory` dict that gives the target GPU a budget and blocks all other GPUs with `"0GiB"`.  CPU offload (`"24GiB"` / `"32GiB"`) as safety net.

---

## Cell 1 — Markdown Title

**Purpose**: Header + pipeline description.
No code executed.

---

## Cell 2 — Install Dependencies

**Purpose**: `%pip install` for langchain, langgraph, transformers, bitsandbytes, etc.

| Section | What |
|---------|------|
| Lightweight | langchain-openai, langgraph, tenacity, accelerate, timm, einops, torchxrayvision, etc. |
| Transformers | `--no-deps "transformers>=4.40.0"` — avoids torch upgrade on Kaggle |
| bitsandbytes | GPU-specific build for NF4/8-bit quantisation |

---

## Cell 3 — Core Imports + Environment Setup

**Key outputs**: paths verified, GPUs listed, logging configured

| Step | What happens |
|------|-------------|
| Standard imports | `torch`, `json`, `re`, `gc`, `sys`, `uuid`, `logging`, `pathlib`, etc. |
| Transformers shim | Patches `find_pruneable_heads_and_indices` and `prune_linear_layer` for `transformers ≥4.45` on Kaggle |
| Package imports | `from medrax_premium.agent import *` — loads Agent, ConflictDetector, ConflictResolver, CanonicalFinding, etc. |
| | `from medrax_premium.tools import *` — loads all 5 tool classes |
| | `from medrax_premium.utils import *` — loads `load_prompts_from_file`, `LazyTool`, `GPUMemoryManager` |
| Kaggle paths | `ROOT`, `BENCH_ROOT`, `MODELS_CHEX`, `MODELS_CORE`, `MODELS_LLAVA`, `MODELS_MEDTOOL` |
| GPU config | `DEVICE_0 = "cuda:0"`, `DEVICE_1 = "cuda:1"` |
| OOM prevention | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Logging | Per-run JSON log to `LOG_DIR`, timestamped |

**Import chain** (conflict resolution enters here):
```
medrax_premium.agent.__init__
  ├── Agent, AgentState
  ├── CanonicalFinding, normalize_output, extract_pathologies_with_polarity
  ├── ConflictDetector, ConflictResolver, Conflict
  ├── ArgumentGraph, ArgumentGraphBuilder
  ├── ToolTrust, ToolTrustManager
  ├── AbstentionLogic, AbstentionDecision
  └── ConfidenceScoringPipeline, ConfidenceFusion
```

---

## Cell 4 — GPU Utilities

**Purpose**: Inline helper functions used by the tool-loading cell (Cell 5).

| Function | What it does |
|----------|-------------|
| `parse_gpu_id(device)` | Extracts integer GPU index from `"cuda:1"` → `1` |
| `check_gpu_free_space(device, required_gb, label)` | Raises `RuntimeError` if target GPU lacks `required_gb` free VRAM |
| `gpu_summary()` | One-liner per GPU: used / total |
| `validate_checkpoint(path, label, required_globs)` | Verifies model weight directory exists + optional file-pattern checks |
| `compute_max_memory(device, headroom_gb, cpu_gb)` | Builds `max_memory` dict that pins quantised model to target GPU only |
| `check_4bit_overflow(device, est_4bit_gb, label)` | Checks if 4-bit model fits; returns `(needs_overflow, free_gb)` |

---

## Cell 5 — Tool Loading (4-Phase) + Agent + Runner

### Phase 0 — Validate Model Checkpoints

Calls `validate_checkpoint()` for every model directory:

| Checkpoint | Required globs |
|-----------|---------------|
| `MODELS_CORE/swinv2_findings` | `*.safetensors`, `config.json` |
| `MODELS_CORE/swinv2_impression` | `*.safetensors`, `config.json` |
| `MODELS_LLAVA` (LLaVA-Med-7B) | `config.json` |
| `MODELS_CHEX` (CheXagent-8B) | `config.json` |

### Phase 1 — GPU-0 Lightweight Tools

```python
check_gpu_free_space(DEVICE_0, required_gb=0.8)
report_tool              = ChestXRayReportGeneratorTool(device=DEVICE_0, ...)  # SwinV2×2, fp16
xray_classification_tool = ChestXRayClassifierTool(device=DEVICE_0)           # DenseNet, fp32
segmentation_tool        = ChestXRaySegmentationTool(device=DEVICE_0)         # PSPNet, fp32
```

### Phase 2 — GPU-0 LLaVA-Med 7B (4-bit NF4)

```python
check_gpu_free_space(DEVICE_0, required_gb=4.5)
check_4bit_overflow(DEVICE_0, est_4bit_gb=4.5)
llava_med_tool = LlavaMedTool(device=DEVICE_0, load_in_4bit=True, ...)
```

**Inside `LlavaMedTool.__init__`** → calls `builder.load_pretrained_model`:
- `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")`
- `device_map="auto"` + `max_memory={0: "<free-1>GiB", 1: "0GiB", "cpu": "24GiB"}`
- Vision tower + mm_projector explicitly moved to `device` after load

### Phase 3 — GPU-1 CheXagent 8B (4-bit NF4)

```python
# Monkey-patch: inject torch_dtype=bf16 so non-quantised layers
# do not materialise in fp32 (~16 GB spike → ~5 GB instead)
_AMCLM.from_pretrained = _patched_from_pretrained   # adds setdefault(torch_dtype=bf16)

check_gpu_free_space(DEVICE_1, required_gb=5.0)
check_4bit_overflow(DEVICE_1, est_4bit_gb=5.0)
xray_vqa_tool = XRayVQATool(device=DEVICE_1, load_in_4bit=True, ...)

_AMCLM.from_pretrained = _real_from_pretrained       # restore
```

**Inside `XRayVQATool.__init__`**:
- `_check_gpu_free_space(device, required_gb=5.0)` (library-side guard)
- `_download_and_patch_model()` → patches `assert transformers.__version__` in CheXagent remote code
- `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")`
- `device_map="auto"` + `max_memory={1: "<budget>GiB", 0: "0GiB", "cpu": "32GiB"}`
  - `budget = min(free - 1.0, total * 0.55)` → keeps headroom for quantisation staging

### Phase 4 — Cleanup + Summary

```python
gc.collect(); torch.cuda.empty_cache()
ALL_TOOLS = [report_tool, xray_classification_tool, segmentation_tool, llava_med_tool, xray_vqa_tool]
# Prints final VRAM usage on both GPUs
```

### `get_agent()` → Premium Agent with conflict resolution

```python
Agent(
    model=ChatOpenAI("gpt-4o-mini"),    # orchestrator
    enable_conflict_resolution=True,     # activates full pipeline
    conflict_sensitivity=0.4,            # ConflictDetector threshold
    deferral_threshold=0.6,              # AbstentionLogic threshold
)
```

**Inside `Agent.__init__`**:
1. `ConflictDetector(sensitivity=0.4)` — BERT NLI + Rule-Based + GACL
2. `ConflictResolver(deferral_threshold=0.6)` — Argumentation Graph + Tool Trust + Abstention
3. `StateGraph`: `process → execute → process → ... → END`
4. Each call gets a fresh `thread_id = uuid4()` (no cross-case state leakage)

### `run_medrax(agent, thread, prompt, image_urls)`

Sends one `HumanMessage` (text + images) through the ReAct workflow. Returns `(response_text, agent_state_str)`.

---

## Cell 6 — Question Processing + `main()`

### `_image_to_data_uri(path)` → base64 data URI

Converts local ChestAgentBench images to inline data URIs so GPT-4o-mini can see them.

### `_extract_letter(text)` → single letter A–F

Robust 5-stage regex extraction:
1. Single-char match
2. Leading letter with word boundary
3. Pattern: "answer is X" / "option X" / "choose X"
4. Parenthesized: `(X)`
5. Standalone word-boundary letter

### `create_multimodal_request(...)` — 2-turn evaluation

```
Turn 1: "Answer this question using chain of thought..." + images
         ↓
    ReAct loop executes (tools fire, conflicts detected/resolved)
         ↓
Turn 2: "Only respond with the letter of choice (A–F)"
         ↓
    _extract_letter() → final answer
```

**Inside the ReAct loop** (`agent.py` → `execute_tools`):

| Step | Component | What happens |
|------|-----------|-------------|
| 1 | Tool invocation | Each tool called, raw result captured |
| 2 | `normalize_output()` | Converts raw output → `CanonicalFinding` objects (pathology, confidence, region, evidence_type) |
| 3 | `_cross_reference_findings()` | Aligns VQA free-text with classifier pathology names — creates synthetic findings so conflicts can be detected across tool types |
| 4 | `ConflictDetector.detect_conflicts()` | Runs 3 detectors in parallel: **BERT NLI** (semantic), **Rule-Based** (presence contradictions), **GACL** (graph-based consistency) |
| 5 | `ConflictResolver.resolve_conflict()` | For each conflict: builds **Argumentation Graph** → applies **Tool Trust** weights → **Abstention Logic** decides: resolve or defer |
| 6 | Append to ToolMessage | Conflict summary injected into last tool result → LLM reads it when generating final answer |
| 7 | `_save_tool_calls_with_conflicts()` | Full JSON log: tool calls + canonical findings + conflicts + resolutions |

### `main()` — benchmark runner

- Loads `metadata.jsonl` from ChestAgentBench
- Groups questions by `case_id`, creates fresh agent per case
- **Resumable**: reads existing `premium_benchmark.jsonl`, skips completed questions
- Tracks: correct/total/skipped/errors, prints running accuracy
- `gc.collect()` + `del agent` after each case (T4 memory management)

---

## Cell 7 — Accuracy Comparison + Figures

**Reads**: `PREMIUM_LOG` (`premium_benchmark.jsonl`)

| Output | Description |
|--------|-------------|
| Comparison table | 9 categories × 6 models (LLaVA-Med, CheXagent, Llama-3.2-90B, GPT-4o, MedRAX, **MedRAX Premium**) |
| `fig_category_comparison.pdf/png` | Grouped bar chart — accuracy per category |
| `fig_overall_accuracy.pdf/png` | Horizontal bar chart — overall accuracy |
| `table_accuracy_comparison.json` | Machine-readable accuracy data |
| `raw_results.json` | Full results dump |

**Categories**: Detection, Classification, Localization, Comparison, Relationship, Diagnosis, Characterization, Enumeration, Reasoning

---

## Cell 8 — Conflict Resolution Statistics

**Reads**: `conflict_resolution_*.json` from `LOG_DIR/tool_logs/`

### Table 1 — Conflict Detection (tool-wise)

| Column | Source |
|--------|--------|
| Semantic | BERT NLI detector |
| Rule-Based | Presence contradiction detector |
| GACL | Graph-based anatomical consistency |

Rows = tool pairs (e.g. "chest_xray_classifier vs chest_xray_expert")

### Table 2 — Conflict Resolution (tool-wise)

Same structure — counts how many conflicts from each detector were resolved.

### Table 3 — Cycle Detection & Abstention

| Metric | What it measures |
|--------|-----------------|
| Argumentation Cycles | Circular reasoning in the argument graph |
| Human Review Flagged | Confidence below `deferral_threshold` |
| Abstention reasons | Breakdown by reason (low confidence, contradictory evidence, etc.) |
| Severity | critical / moderate / minor |

**Saved to**: `table_conflict_resolution.json`

---

## Cell 9 — ACL Submission ZIP

**Writes**: `medrax_premium_acl_submission.zip`

Contents:
```
tables/
  table_accuracy_comparison.json
  table_conflict_resolution.json
  experiment_config.json
  raw_results.json
figures/
  fig_category_comparison.pdf
  fig_category_comparison.png
  fig_overall_accuracy.pdf
  fig_overall_accuracy.png
logs/
  premium_benchmark.jsonl
```

---

## ReAct Loop — Original MedRAX vs MedRAX Premium

Both systems use the same LangGraph `StateGraph` topology:

```
entry → process ──┬── has_tool_calls? ── True  → execute → process (loop)
                  └── False → END
```

### Original MedRAX ReAct Loop (`MedRAX/medrax/agent/agent.py`)

```
┌─────────────────────────────────────────────────┐
│                  process_request                 │
│  Prepend SystemMessage → model.invoke(messages)  │
│  Return AIMessage (may contain tool_calls)        │
└──────────────────────┬──────────────────────────┘
                       │
              has_tool_calls?
              /             \
           True            False → END
            │
┌───────────┴─────────────────────────────────────┐
│                  execute_tools                   │
│                                                  │
│  for each tool_call:                             │
│    1. Validate tool name exists                  │
│    2. tool.invoke(args)                          │
│    3. Wrap raw result as ToolMessage(content=str)│
│                                                  │
│  _save_tool_calls() → JSON log                   │
│  Return ToolMessages                             │
└──────────────────────┬──────────────────────────┘
                       │
                  back to process
```

**Key characteristics:**
- No output normalisation — raw `str(result)` goes straight into ToolMessage
- No conflict detection — LLM sees all tool outputs and must reconcile contradictions on its own
- No token-budget trimming — works because GPT-4o has 128k context
- No error-result detection — errors appear as regular tool output
- Simple JSON log: `tool_calls_{timestamp}.json`

### MedRAX Premium ReAct Loop (`Medrax_premium/medrax_premium/agent/agent.py`)

```
┌─────────────────────────────────────────────────────────┐
│                    process_request                       │
│  Prepend SystemMessage                                   │
│  _trim_messages() ← token-budget (8k limit for free-tier)│
│  model.invoke(messages)                                  │
│    └─ 413 retry: emergency trim, strip ALL images        │
│  Return AIMessage                                        │
└───────────────────────────┬─────────────────────────────┘
                            │
                   has_tool_calls?
                   /              \
                True             False → END
                 │
┌────────────────┴────────────────────────────────────────┐
│                    execute_tools                         │
│                                                          │
│  for each tool_call:                                     │
│    1. Validate tool name                                 │
│    2. tool.invoke(args) with try/except + timing         │
│    3. Error-result detection (check for {"error": ...})  │
│    4. normalize_output() → List[CanonicalFinding]  ← NEW │
│       {pathology, region, confidence, evidence_type,     │
│        source_tool, raw_value}                           │
│    5. _format_tool_result() → compact LLM-readable text  │
│    6. Wrap as ToolMessage                                │
│                                                          │
│  ──── Post-loop pipeline (NEW) ────                      │
│                                                          │
│  7. _cross_reference_findings()                          │
│     VQA text ↔ classifier pathology alignment            │
│     Creates synthetic findings with inverted confidence  │
│                                                          │
│  8. ConflictDetector.detect_conflicts()                  │
│     ├── BERT NLI (semantic entailment/contradiction)     │
│     ├── Rule-Based (presence vs absence for same patho)  │
│     └── GACL (graph-based anatomical consistency)        │
│     Requires findings from 2+ distinct tools             │
│                                                          │
│  9. ConflictResolver.resolve_conflict() [per conflict]   │
│     ├── ArgumentGraphBuilder → directed evidence graph   │
│     ├── Cycle detection (circular reasoning)             │
│     ├── ToolTrustManager → historical accuracy weights   │
│     ├── ConfidenceFusion → weighted combination          │
│     └── AbstentionLogic → defer if confidence < 0.6     │
│                                                          │
│  10. Inject conflict summary into last ToolMessage       │
│      "--- CONFLICT ANALYSIS ---\n⚠️ N conflicts..."     │
│      LLM sees this in next process() call                │
│                                                          │
│  11. _save_tool_calls_with_conflicts() → comprehensive   │
│      JSON: tools + canonical + conflicts + resolutions   │
│                                                          │
│  Return ToolMessages (with conflict annotation)          │
└────────────────────────────┬────────────────────────────┘
                             │
                        back to process
```

### Side-by-Side Comparison

| Aspect | Original MedRAX | MedRAX Premium |
|--------|----------------|----------------|
| **Graph topology** | `process → execute → process` | Same — identical `StateGraph` |
| **LLM backbone** | GPT-4o (128k context) | GPT-4o-mini (8k free-tier) |
| **Token management** | None (128k is enough) | `_trim_messages()` with 5-stage budget + 413 emergency retry |
| **Tool output format** | `str(result)` raw | `_format_tool_result()` — compact, type-aware (classifier → top findings, VQA → expert answer, segmentation → metrics summary) |
| **Output normalisation** | None | `normalize_output()` → `CanonicalFinding` (pathology, confidence, region, evidence_type) |
| **Cross-referencing** | None | `_cross_reference_findings()` — aligns VQA free-text mentions with classifier pathology names |
| **Conflict detection** | None — LLM must reconcile | 3 parallel detectors: BERT NLI, Rule-Based, GACL |
| **Conflict resolution** | None | Argumentation Graph + Tool Trust + Abstention Logic |
| **Human deferral** | None | `should_defer=True` when confidence < 0.6 |
| **Conflict feedback** | None | Injected into ToolMessage → LLM reads conflict summary |
| **Error handling** | Bare `invoke()` | `try/except` + error-dict detection + timing |
| **Logging** | `tool_calls_{ts}.json` (calls only) | `conflict_resolution_{ts}.json` (calls + canonical + conflicts + resolutions) |
| **Image handling** | Passed through | Older images stripped; latest kept; base64 managed for token budget |

### Correctness Verification

Both loops share the same LangGraph wiring:
```python
workflow = StateGraph(AgentState)
workflow.add_node("process", self.process_request)
workflow.add_node("execute", self.execute_tools)
workflow.add_conditional_edges("process", self.has_tool_calls, {True: "execute", False: END})
workflow.add_edge("execute", "process")
workflow.set_entry_point("process")
self.workflow = workflow.compile(checkpointer=checkpointer)
```

**Verified correct:**
- `process → has_tool_calls?` — conditional edge routes to `execute` (tools requested) or `END` (final answer)
- `execute → process` — unconditional edge loops back so LLM can see tool results
- `MemorySaver` checkpointer enables multi-turn within a thread (each benchmark case gets fresh `thread_id`)
- Premium's added stages (normalise, cross-ref, detect, resolve, inject) all happen **inside** `execute_tools` before returning `ToolMessages` — they don't alter the graph topology

---

## Conflict Resolution Flow (Detailed)

```
Tool outputs (raw)
      │
      ▼
normalize_output()          ← canonical_output.py
      │  Converts each tool's raw format to CanonicalFinding:
      │  {pathology, region, confidence, evidence_type, source_tool, raw_value}
      │
      ▼
_cross_reference_findings() ← agent.py
      │  VQA says "no cardiomegaly" → creates synthetic finding
      │  for classifier's "Cardiomegaly" pathology with inverted confidence
      │
      ▼
ConflictDetector.detect_conflicts()  ← conflict_resolution.py
      │  Three parallel detectors:
      │  ├── BERT NLI:  semantic entailment/contradiction between findings
      │  ├── Rule-Based: presence vs absence for same pathology
      │  └── GACL:      anatomical consistency graph checks
      │  Returns: List[Conflict] with severity + tools_involved
      │
      ▼
ConflictResolver.resolve_conflict()  ← conflict_resolution.py
      │  For each conflict:
      │  ├── ArgumentGraphBuilder → builds directed graph of evidence
      │  ├── Cycle detection → flags circular reasoning
      │  ├── ToolTrustManager → weights by historical accuracy
      │  ├── ConfidenceFusion → weighted combination
      │  └── AbstentionLogic → defer if confidence < 0.6
      │  Returns: {decision, confidence, should_defer, abstention_reason, argumentation_graph}
      │
      ▼
Injected into ToolMessage   ← agent.py execute_tools
      │  "--- CONFLICT ANALYSIS ---\n⚠️ 2 conflicts detected..."
      │  LLM sees this alongside tool results in next process() call
      │
      ▼
LLM generates informed answer
      │  GPT-4o-mini uses conflict resolution to weight its response
      │
      ▼
Saved to conflict_resolution_*.json  ← agent.py _save_tool_calls_with_conflicts()
      │  Full audit trail for Cell 8 analysis
```

---

## Key Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `conflict_sensitivity` | 0.4 | Lower = more conflicts detected (more conservative) |
| `deferral_threshold` | 0.6 | Below this confidence → flag for human review |
| `temperature` | 0.2 | Low randomness for reproducible answers |
| `MAX_QUESTIONS` | 100 | Benchmark cap |
| `load_in_4bit` | True | CheXagent + LLaVA-Med quantized (NF4, double-quant) to fit T4 |

---

## Token-Budget Strategy (Agent)

GitHub Models free-tier GPT-4o-mini has an 8 000 input-token hard cap.
Tool schemas (~5 tools) consume ~1 500–2 000 tokens invisibly.

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_TOOL_MSG_CHARS` | 400 | Keep each ToolMessage compact |
| `MAX_AI_MSG_CHARS` | 600 | Trim long assistant reasoning |
| `MAX_TEXT_CHARS` | 12 000 | Total text budget (~3 000 tok) |
| `MAX_MESSAGES` | 12 | Hard cap on conversation turns |

**Trimming strategy** (`_trim_messages`):
1. Keep SystemMessage + newest HumanMessage (with images) intact
2. Strip base64 images from older HumanMessages
3. Truncate ToolMessage/AIMessage contents
4. Drop older middle messages if still over budget
5. Emergency 413 retry: strip ALL images, keep last 6 messages

---

## Files Modified (Source-Level Fixes)

| File | Fix | Why |
|------|-----|-----|
| `medrax_premium/llava/model/builder.py` | `max_memory` uses actual free VRAM (`mem_get_info`) not total | Prevents OOM when GPU-0 already holds lightweight tools |
| `medrax_premium/llava/model/builder.py` | Prints `[LLaVA] max_memory = ...` for debugging | Visibility into accelerate's budget |
| `medrax_premium/tools/xray_vqa.py` | `input_ids.to(device=next(self.model.parameters()).device)` | `device_map="auto"` may place input embeddings differently from raw device string |
| `medrax_premium/tools/xray_vqa.py` | `required_gb=5.0` (was 5.5) for 4-bit guard | Actual NF4 footprint is ~5 GB, not 5.5 |
| `medrax_premium/tools/xray_vqa.py` | `from_list_format` fallback | Newer tokenizers dropped Qwen-era method |
| `medrax_premium/**/*.py` (11 files) | `from medrax.` → `from medrax_premium.` | Package renamed |

---

## Device-Handling Audit (per tool)

| Tool | `__init__` device handling | `_run` tensor placement | Status |
|------|-------------------------|------------------------|:------:|
| Report Generator | `self.device = torch.device(device)` → `.to(self.device)` both models | `pixel_values.to(self.device)` | ✅ |
| Classifier | `self.device = torch.device(device)` → `model.to(self.device)` | `img.to(self.device)` | ✅ |
| Segmentation | `self.device = torch.device(device)` → `model.to(self.device)` | `img.to(self.device)` | ✅ |
| LLaVA-Med | `device_map="auto"` + `max_memory` (quantised) | `input_ids.to(device=self.model.device)`, `image_tensor.to(device=self.model.device)` | ✅ |
| CheXagent | `device_map="auto"` + `max_memory` (quantised) | `input_ids.to(device=next(self.model.parameters()).device)` | ✅ |

**Key insight**: For `device_map="auto"` models, raw device strings (e.g. `"cuda:1"`) can't be used for tensor placement because accelerate may place the input embedding layer on a different physical device. Both quantised tools use model-introspection:
- LLaVA-Med: `self.model.device`
- CheXagent: `next(self.model.parameters()).device`

---

## Max-Memory Budget Computation

### LLaVA-Med (builder.py)

```python
free_gb = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3     # actual free VRAM
budget  = max(4, int(free_gb - 1.0))                        # 1 GB headroom
max_mem = {gpu_id: f"{budget}GiB", other: "0GiB", "cpu": "24GiB"}
```

### CheXagent (xray_vqa.py)

```python
tot    = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
free   = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3
budget = min(free - 1.0, tot * 0.55)                        # conservative: 55% of total cap
max_mem = {gpu_id: f"{max(4, int(budget))}GiB", other: "0GiB", "cpu": "32GiB"}
```

Both emit a print line showing the final `max_memory` dict for debugging.

---

## Monkey-Patch: CheXagent bf16 Override

CheXagent's `trust_remote_code=True` loading can materialise non-quantised layers in fp32, spiking VRAM to ~16 GB before quantisation takes effect.

**The fix** (notebook Cell 5, Phase 3):
```python
_real = AutoModelForCausalLM.from_pretrained
def _patched(path, *a, **kw):
    kw.setdefault("torch_dtype", torch.bfloat16)
    kw.setdefault("low_cpu_mem_usage", True)
    return _real(path, *a, **kw)
AutoModelForCausalLM.from_pretrained = _patched
# ... load CheXagent ...
AutoModelForCausalLM.from_pretrained = _real   # restore immediately
```

`setdefault` ensures it doesn't override if the library code already sets `torch_dtype`.

---

## Tool Selection Analysis (9 tools in original MedRAX)

| # | Tool Class | Model / Backend | GPU / Memory | Used? | Reason |
|---|-----------|----------------|-------------|:---:|--------|
| 1 | `ChestXRayReportGeneratorTool` | SwinV2 (CheXpert Plus) | GPU-0, ~0.5 GB fp16 | ✅ | Generates findings + impression reports — directly answers report-based MCQs |
| 2 | `ChestXRayClassifierTool` | DenseNet-121 (TorchXRayVision) | GPU-0, ~0.03 GB fp32 | ✅ | Classifies 18 pathologies — essential for detection/classification MCQs |
| 3 | `ChestXRaySegmentationTool` | PSPNet (ChestX-Det) | GPU-0, ~0.1 GB fp32 | ✅ | Segments 14 anatomical structures with area metrics — useful for comparison/relationship MCQs |
| 4 | `LlavaMedTool` | LLaVA-Med 7B (NF4 4-bit) | GPU-0, ~4.5 GB | ✅ | General biomedical VQA — complements CheXagent; enables conflict-resolution cross-validation |
| 5 | `XRayVQATool` | CheXagent 8B (NF4 4-bit) | GPU-1, ~5.0 GB | ✅ | Expert-level chest X-ray VQA — strongest single tool for diverse medical questions |
| 6 | `XRayPhraseGroundingTool` | MAIRA-2 (8-bit) | ~8 GB (not loaded) | ❌ | **Gated HuggingFace repo** → 401 on Kaggle; draws bounding boxes — irrelevant to MCQ letter choices |
| 7 | `ChestXRayGeneratorTool` | RoentGen (Stable Diffusion) | ~4 GB (not loaded) | ❌ | **Manual weight download** (contact Stanford MIMI); generates synthetic X-rays — irrelevant to MCQs |
| 8 | `DicomProcessorTool` | None (pydicom CPU) | Negligible | ❌ | Converts DICOM → PNG; ChestAgentBench images are **already PNG** — tool would never be invoked |
| 9 | `ImageVisualizerTool` | None (matplotlib CPU) | Negligible | ❌ | Calls `plt.show()` for display; **headless Kaggle** has no display — dead code |

### Summary

| | Count | Tools | GPU |
|---|:---:|-------|-----|
| **Used** | 5 | Report, Classification, Segmentation, LLaVA-Med, CheXagent | GPU-0 + GPU-1 |
| **Skipped** | 4 | Grounding (MAIRA-2), Generation (RoentGen), DICOM Processor, Image Visualizer | — |

**Why these 5?** — All five provide analytical capabilities that directly answer MCQs. Together they cover every ChestAgentBench category (Detection, Classification, Localization, Comparison, Relationship, Diagnosis, Characterization). Two VQA tools on separate GPUs enable the Premium conflict-resolution system to cross-validate answers.

**Why not the other 4?**
- **Grounding & Generation**: Inaccessible on Kaggle (gated repo / manual weights) and functionally irrelevant to MCQ answering.
- **DICOM & Visualizer**: Zero-cost utilities but add unnecessary tool options that could mislead the agent into wasting reasoning steps on no-op actions during headless benchmark evaluation.
