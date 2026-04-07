<h1 align="center">
🤖 MedRAX: Medical Reasoning Agent for Chest X-ray
</h1>
<p align="center"> <a href="https://arxiv.org/abs/2502.02673" target="_blank"><img src="https://img.shields.io/badge/arXiv-ICML 2025-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a> <a href="https://github.com/bowang-lab/MedRAX"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a> <a href="https://huggingface.co/datasets/wanglab/chest-agent-bench"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Dataset"></a> </p>

![](assets/demo_fast.gif?autoplay=1)

<br>

## 📋 Table of Contents

- [📋 Table of Contents](#-table-of-contents)
- [Abstract](#abstract)
- [MedRAX Overview](#medrax-overview)
  - [Integrated Tools](#integrated-tools)
- [ChestAgentBench](#chestagentbench)
- [Advanced: Conflict Detection \& Resolution Pipeline](#advanced-conflict-detection--resolution-pipeline)
  - [Pipeline Architecture](#pipeline-architecture)
  - [Layer 1: Conflict Detection](#layer-1-conflict-detection)
    - [Stage 1: Deduplication](#stage-1-deduplication)
    - [Stage 2: Tool-Level Comparison (Primary Detection)](#stage-2-tool-level-comparison-primary-detection)
    - [Stage 3: BERT-NLI Semantic Enrichment](#stage-3-bert-nli-semantic-enrichment)
    - [Anatomical Consistency Module (GACL)](#anatomical-consistency-module-gacl)
  - [Layer 2: Confidence Calibration and Fusion](#layer-2-confidence-calibration-and-fusion)
  - [Layer 3: Argumentation Graphs](#layer-3-argumentation-graphs)
  - [Layer 4: Tool Trust Management](#layer-4-tool-trust-management)
  - [Layer 5: Intelligent Abstention](#layer-5-intelligent-abstention)
  - [End-to-End Resolution Flow](#end-to-end-resolution-flow)
  - [Logging and Traceability](#logging-and-traceability)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
  - [Getting Started](#getting-started)
- [Tool Selection and Initialization](#tool-selection-and-initialization)
- [Automatically Downloaded Models](#automatically-downloaded-models)
  - [Classification Tool](#classification-tool)
  - [Segmentation Tool](#segmentation-tool)
  - [Grounding Tool](#grounding-tool)
  - [LLaVA-Med Tool](#llava-med-tool)
  - [Report Generation Tool](#report-generation-tool)
  - [Visual QA Tool](#visual-qa-tool)
  - [MedSAM Tool](#medsam-tool)
  - [Utility Tools](#utility-tools)
- [Manual Setup Required](#manual-setup-required)
  - [Image Generation Tool](#image-generation-tool)
- [Configuration Notes](#configuration-notes)
  - [Required Parameters](#required-parameters)
  - [Memory Management](#memory-management)
  - [Local LLMs](#local-llms)
  - [Optional: OpenAI-compatible Providers](#optional-openai-compatible-providers)
- [Star History](#star-history)
- [Authors \& Citation](#authors--citation)
  - [Authors](#authors)
- [Citation](#citation)

<br>

## Abstract
Chest X-rays (CXRs) play an integral role in driving critical decisions in disease management and patient care. While recent innovations have led to specialized models for various CXR interpretation tasks, these solutions often operate in isolation, limiting their practical utility in clinical practice. We present MedRAX, the first versatile AI agent that seamlessly integrates state-of-the-art CXR analysis tools and multimodal large language models into a unified framework. MedRAX dynamically leverages these models to address complex medical queries without requiring additional training. To rigorously evaluate its capabilities, we introduce ChestAgentBench, a comprehensive benchmark containing 2,500 complex medical queries across 7 diverse categories. Our experiments demonstrate that MedRAX achieves state-of-the-art performance compared to both open-source and proprietary models, representing a significant step toward the practical deployment of automated CXR interpretation systems.
<br><br>


## MedRAX Overview

MedRAX is built on a robust technical foundation:
- **Core Architecture**: Built on LangChain and LangGraph frameworks
- **Language Model**: Uses GPT-4o with vision capabilities as the backbone LLM
- **Deployment**: Supports both local and cloud-based deployments
- **Interface**: Production-ready interface built with Gradio
- **Modular Design**: Tool-agnostic architecture allowing easy integration of new capabilities

### Integrated Tools
- **Visual QA**: Utilizes CheXagent and LLaVA-Med for complex visual understanding and medical reasoning
- **Segmentation**: Employs MedSAM and PSPNet model trained on ChestX-Det for precise anatomical structure identification
- **Grounding**: Uses Maira-2 for localizing specific findings in medical images
- **Report Generation**: Implements SwinV2 Transformer trained on CheXpert Plus for detailed medical reporting
- **Disease Classification**: Leverages DenseNet-121 from TorchXRayVision for detecting 18 pathology classes
- **X-ray Generation**: Utilizes RoentGen for synthetic CXR generation
- **Utilities**: Includes DICOM processing, visualization tools, and custom plotting capabilities
<br><br>


## ChestAgentBench
We introduce ChestAgentBench, a comprehensive evaluation framework with 2,500 complex medical queries across 7 categories, built from 675 expert-curated clinical cases. The benchmark evaluates complex multi-step reasoning in CXR interpretation through:

- Detection
- Classification
- Localization
- Comparison
- Relationship
- Diagnosis
- Characterization

Download the benchmark: [ChestAgentBench on Hugging Face](https://huggingface.co/datasets/wanglab/chest-agent-bench)
```
huggingface-cli download wanglab/chestagentbench --repo-type dataset --local-dir chestagentbench
```

Unzip the Eurorad figures to your local `MedMAX` directory.
```
unzip chestagentbench/figures.zip
```

To evaluate with GPT-4o, set your OpenAI API key and run the quickstart script.
```
export OPENAI_API_KEY="<your-openai-api-key>"
python quickstart.py \
    --model chatgpt-4o-latest \
    --temperature 0.2 \
    --max-cases 2 \
    --log-prefix chatgpt-4o-latest \
    --use-urls
```

<br>

## Advanced: Conflict Detection & Resolution Pipeline

When multiple specialized AI models interpret the same chest X-ray, their outputs frequently diverge. A DenseNet-121 classifier may report cardiomegaly with 89% confidence while CheXagent reports its absence at 75% confidence, and LLaVA-Med generates a free-text report describing "borderline cardiac enlargement." These disagreements arise from fundamental differences in training data distributions, architectural inductive biases, task-specific optimization objectives, and uncalibrated confidence scales across heterogeneous model families. In a clinical setting, unresolved contradictions of this nature pose a direct risk to patient safety.

MedRAX addresses this challenge through a five-layer conflict detection and resolution pipeline that systematically identifies disagreements, normalizes heterogeneous confidence scores, structures competing interpretations as formal argumentation graphs, learns tool reliability over time, and knows when to abstain and defer to human expertise rather than force an unreliable decision.

### Pipeline Architecture

The pipeline processes tool outputs sequentially through five layers, where each layer refines the decision boundary established by the previous one:

| Layer | Component | Function | Core Technique |
|:-----:|-----------|----------|----------------|
| 1 | **Conflict Detection** | Identify disagreements across tool outputs | Tool-level pairwise comparison + BERT-NLI enrichment |
| 2 | **Confidence Calibration** | Normalize scores to a common probabilistic scale | Isotonic regression, temperature scaling, min-max normalization |
| 3 | **Argumentation Graphs** | Structure conflicts as weighted support/attack graphs | Bipolar argumentation frameworks with cycle detection |
| 4 | **Tool Trust Management** | Weight opinions by historically learned reliability | Bayesian performance tracking with persistent state |
| 5 | **Intelligent Abstention** | Refuse to decide when evidence is insufficient | Risk-aware thresholds stratified by clinical severity |

The final output of the pipeline is a resolved finding accompanied by a calibrated confidence score, an explicit reasoning trace, and — when applicable — a deferral flag indicating that the case requires human radiologist review.

---

### Layer 1: Conflict Detection

The first layer uses a **tool-level pairwise comparison** strategy that guarantees at most $\binom{N}{2}$ conflicts when $N$ distinct tools are invoked. This bound prevents the combinatorial explosion that would occur if individual findings were compared pairwise (e.g., three classifier runs on three images producing 54 findings and hundreds of spurious conflicts). Detection proceeds in three stages:

#### Stage 1: Deduplication

When the same tool is called multiple times on different images (e.g., DenseNet-121 on three views of the same case), it produces multiple findings for each pathology. The detector retains only the highest-confidence entry per (tool, pathology) pair, collapsing redundant entries before any comparison occurs. Synthetic VQA cross-reference findings are calibrated: positive mentions are floored at 0.55 confidence and negative mentions are capped at 0.20, preventing uncalibrated VQA base-confidence values from triggering false conflicts.

#### Stage 2: Tool-Level Comparison (Primary Detection)

For each unordered pair of distinct tools, the detector identifies shared pathologies — those where both tools have reported an opinion. Among the shared pathologies, it finds the one with the **worst present-vs-absent disagreement**: where one tool's calibrated confidence exceeds 0.5 (indicating presence) and the other falls below 0.5 (indicating absence), and the absolute gap between their confidences is maximal. If this worst-case gap exceeds the configurable sensitivity threshold (default: 0.25), exactly one conflict is registered for that tool pair. The conflict captures the disagreeing pathology, both tools' names and confidence values, and a severity classification: **critical** when the gap exceeds 0.5, **moderate** when it exceeds 0.3, and **minor** otherwise.

This approach ensures that ordinary agreement between tools (which is the common case) produces zero conflicts, while genuine diagnostic disagreements are captured with minimal noise.

#### Stage 3: BERT-NLI Semantic Enrichment

Once a conflict is identified by the rule-based detector, the system enriches it with semantic analysis from a fine-tuned DeBERTa model (Microsoft DeBERTa-base-MNLI), trained on over 433,000 natural language inference examples. The model classifies the textual relationship between the two conflicting findings into three categories:

- **Contradiction**: The statements are logically incompatible (e.g., *"pneumothorax present, occupying 15% of hemithorax"* versus *"no pneumothorax detected"*).
- **Entailment**: The statements express the same finding in different terminology. If BERT reports high entailment (>70%), the downstream resolver can dismiss the conflict as a false positive.
- **Neutral**: The statements address different clinical aspects and neither support nor contradict each other.

The BERT probability distribution (contradiction, entailment, neutral) is attached to the Conflict object and flows into the resolution pipeline, where it influences severity classification — a BERT contradiction probability above 0.85 upgrades the conflict to **critical**, and above 0.70 upgrades **minor** to **moderate** — as well as the BERT-guided resolution strategy and the confidence adjustment applied to the final decision. This layered approach means BERT semantic analysis is always available to the resolver without inflating the conflict count.

#### Anatomical Consistency Module (GACL)

The system also includes a Graph-based Anatomical Consistency Logic (GACL) module that validates whether the combined set of reported findings is physically plausible. Each finding is represented along five universal, disease-agnostic attribute axes:

- **Occupancy** — whether a finding is present or absent
- **Aeration** — the degree of air presence (normal, decreased, absent)
- **Density** — radiodensity category (air, fluid, soft tissue, calcified)
- **Volume Change** — size relative to normal (increased, decreased, normal)
- **Mass Effect** — mechanical impact on surrounding structures (shift, compression, none)

GACL constructs a constraint graph over reported findings and checks for violations — for example, a region simultaneously reported as containing air-density pneumothorax and fluid-density consolidation represents a physical impossibility. This module operates as a supporting consistency check available for cases involving segmentation and classification data.

---

### Layer 2: Confidence Calibration and Fusion

A fundamental challenge in multi-model systems is that raw confidence scores are not directly comparable across models. A 0.85 from DenseNet-121 does not carry the same probabilistic meaning as a 0.85 from Maira-2, because each model's output distribution reflects its own training dynamics, loss function, and calibration characteristics.

MedRAX addresses this through a task-aware calibration pipeline that converts all tool outputs — whether they are classification logits, free-text statements, segmentation masks, or bounding-box scores — into calibrated probabilities on a unified [0, 1] scale.

The calibration pipeline applies three techniques depending on data availability:

**Min-Max Normalization** is applied as a baseline when the theoretical bounds of a tool's output are known, linearly rescaling raw values to the unit interval.

**Isotonic Regression** provides non-parametric calibration by learning a monotonic mapping from raw scores to true probabilities using historically verified resolutions as ground truth. This approach makes no distributional assumptions and is particularly effective for heterogeneous model families where parametric methods (such as Platt scaling) underperform.

**Temperature Scaling** adjusts the sharpness of a model's softmax distribution by dividing logits by a learned temperature parameter *T* before applying the softmax function. When *T* > 1, overconfident predictions are softened; when *T* < 1, predictions are sharpened. This method preserves the ranking of predictions while improving calibration, making it suitable for models that are directionally correct but poorly calibrated.

Once individual scores are calibrated, they are fused into a single per-finding confidence via a trust-weighted average: each tool's calibrated confidence is multiplied by its learned trust weight (see Layer 4), and the weighted sum is normalized by the total weight mass. This produces a single, interpretable confidence value that reflects both the evidence and the reliability of its sources.

---

### Layer 3: Argumentation Graphs

Where Layers 1 and 2 identify *that* a conflict exists and *what* the calibrated scores are, Layer 3 structures *why* the conflict exists and *how* it should be interpreted. MedRAX formalizes each conflict as a bipolar weighted argumentation graph — a construct from computational argumentation theory adapted for multi-model medical reasoning.

For a given clinical claim (e.g., "Pneumothorax is present"), each tool that has reported on this finding is represented as a node in the graph. Nodes are partitioned into two camps:

- **Support nodes**: Tools whose calibrated output endorses the claim (confidence > 0.5)
- **Attack nodes**: Tools whose calibrated output opposes the claim (confidence ≤ 0.5)

Each node carries a **strength** value computed as the product of its calibrated confidence and its trust weight. The aggregate **support strength** is the sum of all support node strengths, and the aggregate **attack strength** is the sum of all attack node strengths. From these, the system computes two critical metrics:

- **Certainty**: A normalized measure of how dominant the winning side is, ranging from 0.0 (completely ambiguous) to 1.0 (unanimous agreement). It is defined as the ratio of the winning side's strength to the total combined strength.
- **Confidence Gap**: The absolute difference between support and attack strengths, used as input to the abstention logic in Layer 5.

The graph builder also performs **cycle detection** to identify cases where tools create circular justification patterns — a pathological condition where, for instance, Tool A's output depends on Tool B's segmentation, which in turn was influenced by Tool A's classification. Detected cycles are flagged as they undermine the independence assumption required for reliable voting.

The resulting argumentation graph provides full explainability: a clinician can inspect exactly which tools supported or opposed a finding, with what confidence, weighted by what trust level, and whether the decision was clear or marginal. This transparency is essential for clinical adoption, where opaque "black-box" decisions are unacceptable.

---

### Layer 4: Tool Trust Management

Not all tools are equally reliable across all clinical scenarios. A DenseNet-121 classifier trained on TorchXRayVision may excel at detecting cardiomegaly but underperform on subtle pneumothorax cases, while CheXagent may show the opposite pattern. MedRAX captures these reliability differences through a persistent trust management system that learns from historical performance.

Each tool maintains a trust record consisting of a cumulative count of correct predictions and total predictions. The trust weight is simply the ratio of correct to total — an empirical accuracy estimate that converges to the tool's true reliability as more cases are processed. Tools begin with a neutral prior (initialized at 1.0 with a pseudocount of 10 observations to avoid cold-start instability) and are updated after each resolved conflict based on whether their prediction aligned with the final resolution.

Trust weights are persisted to disk as a JSON file and survive application restarts, enabling the system to accumulate institutional knowledge over time. In a deployment scenario, a hospital running MedRAX continuously would observe its trust weights converging to reflect the actual performance characteristics of each tool on their specific patient population and imaging equipment — a form of implicit domain adaptation without any model retraining.

These weights directly influence two critical pipeline stages: they modulate the strength of each tool's node in the argumentation graph (Layer 3), and they determine the weighting coefficients in confidence fusion (Layer 2). The effect is that tools which have historically been more accurate exert proportionally greater influence on the final decision.

---

### Layer 5: Intelligent Abstention

A clinically safe system must know when *not* to decide. Layer 5 implements a multi-criteria abstention logic that evaluates whether the evidence accumulated through Layers 1–4 is sufficient to warrant an automated resolution, or whether the case should be escalated to a human radiologist.

The abstention module evaluates five conditions, any one of which is sufficient to trigger deferral:

1. **Insufficient Evidence**: Fewer than two tools have reported on the finding in question. A single-tool opinion provides no basis for conflict resolution and cannot be validated.

2. **Circular Reasoning**: The argumentation graph contains cycles, indicating that the tools' outputs are not independent. Resolution under these conditions would be logically unsound.

3. **Close Vote**: The absolute gap between aggregate support strength and aggregate attack strength falls below a configurable threshold (default: 0.2). When evidence is nearly evenly split, any automated decision carries unacceptable uncertainty.

4. **High Uncertainty**: The overall certainty score from the argumentation graph falls below a general threshold (default: 0.6), indicating that even the winning side does not command strong enough evidence.

5. **Critical Finding with Insufficient Certainty**: For findings classified as clinically critical — including pneumothorax, tension pneumothorax, large pleural effusions, and other life-threatening conditions — the certainty threshold is elevated to 0.8. This asymmetric threshold reflects the medical principle that the cost of a false negative on a critical finding far exceeds the cost of a false positive.

When abstention is triggered, the system sets the output confidence to zero, attaches a structured explanation of why abstention occurred, assigns a risk level (low, medium, or high), and flags the case for human review. Importantly, the human expert's subsequent decision is fed back into Layer 4 to update the trust weights of the involved tools, creating a closed learning loop that improves system performance over time.

---

### End-to-End Resolution Flow

To illustrate how these five layers interact in practice, consider a scenario where three tools produce conflicting assessments of cardiomegaly on the same chest X-ray:

The DenseNet-121 classifier reports cardiomegaly as present with 89% confidence. CheXagent reports it as absent with 75% confidence. LLaVA-Med generates a free-text report stating that "the cardiac silhouette is enlarged, consistent with cardiomegaly."

**Layer 1** performs tool-level pairwise comparison. With three tools, there are at most $\binom{3}{2} = 3$ potential conflicts. For the DenseNet-121 vs CheXagent pair, cardiomegaly is the worst disagreement: DenseNet says present at 89% while CheXagent (inverted) says present at only 25% — a 0.64 gap that exceeds the 0.25 sensitivity threshold, producing one **moderate** conflict. BERT-NLI enriches this conflict with a contradiction probability of 0.91, upgrading it to **critical**. The other tool pairs (DenseNet vs LLaVA-Med, CheXagent vs LLaVA-Med) may also produce conflicts on cardiomegaly, but with aligned assessments they may fall below the threshold. The total conflict count is bounded at 3.

**Layer 2** calibrates the raw scores. DenseNet-121's 0.89 is adjusted to 0.87 using its learned calibration curve (the model is known to be slightly overconfident). LLaVA-Med's textual output is mapped to a numerical confidence of 0.72 based on its language patterns. CheXagent's 0.75 confidence in absence is inverted to 0.25 confidence in presence.

**Layer 3** constructs the argumentation graph. DenseNet-121 (strength: 0.87 × 0.92 trust = 0.80) and LLaVA-Med (strength: 0.72 × 0.82 trust = 0.59) form the support camp with aggregate strength 1.39. CheXagent (strength: 0.25 × 0.76 trust = 0.19) forms the attack camp. The certainty score is 0.88, and no cycles are detected.

**Layer 4** confirms that DenseNet-121 carries the highest trust weight (0.92) based on 92 correct predictions out of 100 historical cases for cardiac pathology.

**Layer 5** evaluates abstention criteria: three tools are reporting (sufficient), no cycles exist, the strength gap is 1.20 (well above the close-vote threshold), certainty is 0.88 (above both the general and critical thresholds). All criteria pass — the system proceeds with the resolution.

The final output is: cardiomegaly **present**, calibrated confidence **0.84**, supported by DenseNet-121 and LLaVA-Med, opposed by CheXagent, with full reasoning trace and no human review required.

---

### Logging and Traceability

Every conflict resolution is logged with complete traceability, including the initial tool outputs and confidence scores, the BERT-NLI analysis with full probability distributions, the argumentation graph structure, the trust weights used, the abstention evaluation, and the final resolution with reasoning chain. Logs are persisted as timestamped JSON files, enabling retrospective audit, performance analysis, and continuous improvement of the conflict resolution parameters. This audit trail is essential for regulatory compliance in clinical AI deployment and provides the evidence base needed for institutional trust in automated CXR interpretation.

<br>

## Installation
### Prerequisites
- Python 3.10+
- CUDA/GPU for best performance

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/bowang-lab/MedRAX.git
cd MedRAX

# Install package
pip install -e .
```

### Getting Started
```bash
# Start the Gradio interface
python main.py
```
or if you run into permission issues
```bash
sudo -E env "PATH=$PATH" python main.py
```
You need to setup the `model_dir` inside `main.py` to the directory where you want to download or already have the weights of above tools from Hugging Face.
Comment out the tools that you do not have access to.
Make sure to setup your OpenAI API key in `.env` file!
<br><br><br>


## Tool Selection and Initialization

MedRAX supports selective tool initialization, allowing you to use only the tools you need. Tools can be specified when initializing the agent (look at `main.py`):

```python
selected_tools = [
    "ImageVisualizerTool",
    "ChestXRayClassifierTool",
    "ChestXRaySegmentationTool",
    # Add or remove tools as needed
]

agent, tools_dict = initialize_agent(
    "medrax/docs/system_prompts.txt",
    tools_to_use=selected_tools,
    model_dir="/model-weights"
)
```

<br><br>
## Automatically Downloaded Models

The following tools will automatically download their model weights when initialized:

### Classification Tool
```python
ChestXRayClassifierTool(device=device)
```

### Segmentation Tool
```python
ChestXRaySegmentationTool(device=device)
```

### Grounding Tool
```python
XRayPhraseGroundingTool(
    cache_dir=model_dir, 
    temp_dir=temp_dir, 
    load_in_8bit=True, 
    device=device
)
```
- Maira-2 weights download to specified `cache_dir`
- 8-bit and 4-bit quantization available for reduced memory usage

### LLaVA-Med Tool
```python
LlavaMedTool(
    cache_dir=model_dir, 
    device=device, 
    load_in_8bit=True
)
```
- Automatic weight download to `cache_dir`
- 8-bit and 4-bit quantization available for reduced memory usage

### Report Generation Tool
```python
ChestXRayReportGeneratorTool(
    cache_dir=model_dir, 
    device=device
)
```

### Visual QA Tool
```python
XRayVQATool(
    cache_dir=model_dir, 
    device=device
)
```
- CheXagent weights download automatically

### MedSAM Tool
```
Support for MedSAM segmentation will be added in a future update.
```

### Utility Tools
No additional model weights required:
```python
ImageVisualizerTool()
DicomProcessorTool(temp_dir=temp_dir)
```
<br>

## Manual Setup Required

### Image Generation Tool
```python
ChestXRayGeneratorTool(
    model_path=f"{model_dir}/roentgen", 
    temp_dir=temp_dir, 
    device=device
)
```
- RoentGen weights require manual setup:
  1. Contact authors: https://github.com/StanfordMIMI/RoentGen
  2. Place weights in `{model_dir}/roentgen`
  3. Optional tool, can be excluded if not needed
<br>

## Configuration Notes

### Required Parameters
- `model_dir` or `cache_dir`: Base directory for model weights that Hugging Face uses
- `temp_dir`: Directory for temporary files
- `device`: "cuda" for GPU, "cpu" for CPU-only

### Memory Management
- Consider selective tool initialization for resource constraints
- Use 8-bit quantization where available
- Some tools (LLaVA-Med, Grounding) are more resource-intensive
<br>

### Local LLMs
If you are running a local LLM using frameworks like [Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/), you need to configure your environment variables accordingly. For example:
```
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
```
<br>

### Optional: OpenAI-compatible Providers

MedRAX supports OpenAI-compatible APIs, allowing regional or local LLM providers to serve as alternative backends.

For example, to use **Qwen3-VL** via [Alibaba Cloud DashScope](https://bailian.console.aliyun.com/?tab=model#/model-market), set the following environment variables:

```bash
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_API_KEY="<your-dashscope-api-key>"
export OPENAI_MODEL="qwen3-vl-235b-a22b-instruct"
```
<br>

## Star History
<div align="center">
  
[![Star History Chart](https://api.star-history.com/svg?repos=bowang-lab/MedRAX&type=Date)](https://star-history.com/#bowang-lab/MedRAX&Date)

</div>
<br>


## Authors & Citation

### Authors
- **Adibvafa Fallahpour**¹²³⁴ * (adibvafa.fallahpour@mail.utoronto.ca)
- ****Jun Ma****²³ *
- **Alif Munim**³⁵ *
- ****Hongwei Lyu****³
- ****Bo Wang****¹²³⁶

¹ Department of Computer Science, University of Toronto, Toronto, Canada <br>
² Vector Institute, Toronto, Canada <br>
³ University Health Network, Toronto, Canada <br>
⁴ Cohere, Toronto, Canada <br>
⁵ Cohere Labs, Toronto, Canada <br>
⁶ Department of Laboratory Medicine and Pathobiology, University of Toronto, Toronto, Canada

<br>
* Equal contribution
<br><br>


## Citation
If you find this work useful, please cite our paper:
```bibtex
@misc{fallahpour2025medraxmedicalreasoningagent,
      title={MedRAX: Medical Reasoning Agent for Chest X-ray}, 
      author={Adibvafa Fallahpour and Jun Ma and Alif Munim and Hongwei Lyu and Bo Wang},
      year={2025},
      eprint={2502.02673},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.02673}, 
}
```

---
<p align="center">
Made with ❤️ at University of Toronto, Vector Institute, and University Health Network
</p>
