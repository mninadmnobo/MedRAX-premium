import json
import operator
import time as _time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional, Tuple
import copy
import os

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from .canonical_output import normalize_output, CanonicalFinding, extract_pathologies_with_polarity
from .conflict_resolution import ConflictDetector, ConflictResolver, generate_conflict_report

_ = load_dotenv()


class ToolCallLog(TypedDict):
    """
    A TypedDict representing a log entry for a tool call.

    Attributes:
        timestamp (str): The timestamp of when the tool call was made.
        tool_call_id (str): The unique identifier for the tool call.
        name (str): The name of the tool that was called.
        args (Any): The arguments passed to the tool.
        content (str): The content or result of the tool call.
    """

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests and executes tools based on
    language model responses.

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tools (Dict[str, BaseTool]): A dictionary of available tools.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
        log_tools (bool): Whether to log tool calls.
        log_path (Path): Path to save tool call logs.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
        enable_conflict_resolution: bool = True,
        conflict_sensitivity: float = 0.4,
        deferral_threshold: float = 0.6,
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
            log_tools (bool, optional): Whether to log tool calls. Defaults to True.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs'.
            enable_conflict_resolution (bool, optional): Enable conflict detection/resolution. Defaults to True.
            conflict_sensitivity (float, optional): Conflict detection sensitivity (0-1). Defaults to 0.4.
            deferral_threshold (float, optional): Confidence threshold for human deferral. Defaults to 0.6.
        """
        self.system_prompt = system_prompt
        self.log_tools = log_tools
        self.enable_conflict_resolution = enable_conflict_resolution

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)
        
        # Initialize conflict detection and resolution
        if self.enable_conflict_resolution:
            self.conflict_detector = ConflictDetector(sensitivity=conflict_sensitivity)
            self.conflict_resolver = ConflictResolver(deferral_threshold=deferral_threshold)
            print(f"✅ Conflict resolution enabled (sensitivity={conflict_sensitivity}, deferral={deferral_threshold})")
        
        # Define the agent workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        # Aggregated findings across turns keyed by image path.
        # Used to run conflict detection/resolution across multiple tool runs
        # that may occur in separate agent turns.
        self.aggregated_findings = {}

    def reset_for_new_question(self):
        """Reset per-question state. Call before each new benchmark question."""
        self.aggregated_findings = {}

    # ---- token-budget helpers ------------------------------------------------
    # GitHub Models free-tier GPT-4o has an 8 000 input-token hard cap.
    # Tool schemas (~6-9 tools) consume ~2 000-2 500 tokens invisibly.
    # We budget conservatively: 8000 - 2500 (tools) - 100 (overhead) = 5400 tokens
    # ≈ 21 600 chars at 4 chars/tok.  But images at detail:low cost 85 tok each
    # so we budget in chars for TEXT only and handle images separately.
    MAX_TOOL_MSG_CHARS  = 400        # keep each ToolMessage very compact
    MAX_AI_MSG_CHARS    = 600        # trim long assistant reasoning
    MAX_TEXT_CHARS      = 12_000     # total text budget (~3000 tok), leaves room for tool schemas+images
    MAX_MESSAGES        = 12         # hard cap on conversation turns sent to model

    @staticmethod
    def _truncate_content(content: str, limit: int) -> str:
        if len(content) <= limit:
            return content
        half = limit // 2 - 20
        return content[:half] + "\n…[truncated]…\n" + content[-half:]

    @staticmethod
    def _strip_images_from_human(msg):
        """Return a copy of a HumanMessage with base64 images removed (keeps text)."""
        if not isinstance(msg, HumanMessage):
            return msg
        content = msg.content
        if isinstance(content, list):
            text_parts = [p for p in content if isinstance(p, dict) and p.get("type") == "text"]
            if not text_parts:
                text_parts = [{"type": "text", "text": "[image removed to save tokens]"}]
            new_msg = HumanMessage(content=text_parts)
            return new_msg
        return msg

    def _trim_messages(self, messages):
        """
        Aggressive trimming to fit within GitHub Models 8k token limit.
        
        Strategy:
        1. Keep SystemMessage and the newest HumanMessage intact (with images).
        2. Strip images from OLDER HumanMessages.
        3. Truncate ToolMessage and AIMessage contents.
        4. If still over budget, keep only the last MAX_MESSAGES messages.
        5. Final pass: aggressively shorten if total chars still too high.
        """
        if not messages:
            return messages
        
        # Find the last HumanMessage index (the one we want to keep images on)
        last_human_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break
        
        trimmed = []
        for i, m in enumerate(messages):
            if isinstance(m, SystemMessage):
                trimmed.append(m)
            elif isinstance(m, HumanMessage):
                if i == last_human_idx:
                    trimmed.append(m)  # keep images on latest human msg
                else:
                    trimmed.append(self._strip_images_from_human(m))  # strip images from older ones
            elif isinstance(m, ToolMessage):
                content = self._truncate_content(str(m.content), self.MAX_TOOL_MSG_CHARS)
                trimmed.append(ToolMessage(
                    tool_call_id=m.tool_call_id,
                    name=m.name,
                    content=content,
                ))
            elif isinstance(m, AIMessage):
                # Truncate AI reasoning but preserve tool_calls structure
                new_m = copy.copy(m)
                if isinstance(m.content, str) and len(m.content) > self.MAX_AI_MSG_CHARS:
                    new_m.content = self._truncate_content(m.content, self.MAX_AI_MSG_CHARS)
                trimmed.append(new_m)
            else:
                trimmed.append(m)
        
        # Calculate text size (exclude base64 image data from count)
        def _text_size(msg):
            if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
                return sum(len(p.get("text", "")) for p in msg.content if isinstance(p, dict) and p.get("type") == "text")
            return len(str(msg.content)) if hasattr(msg, 'content') else 0
        
        total = sum(_text_size(m) for m in trimmed)
        
        # If still over budget, drop older middle messages (keep system + last few)
        if total > self.MAX_TEXT_CHARS and len(trimmed) > self.MAX_MESSAGES:
            # Keep system message(s) at front + last MAX_MESSAGES-1
            system_msgs = [m for m in trimmed if isinstance(m, SystemMessage)]
            non_system = [m for m in trimmed if not isinstance(m, SystemMessage)]
            keep = non_system[-(self.MAX_MESSAGES - len(system_msgs)):]
            trimmed = system_msgs + keep
            total = sum(_text_size(m) for m in trimmed)
        
        # Final aggressive pass
        if total > self.MAX_TEXT_CHARS:
            for i, m in enumerate(trimmed):
                if total <= self.MAX_TEXT_CHARS:
                    break
                if isinstance(m, ToolMessage):
                    old_len = len(str(m.content))
                    trimmed[i] = ToolMessage(
                        tool_call_id=m.tool_call_id,
                        name=m.name,
                        content=self._truncate_content(str(m.content), 150),
                    )
                    total -= old_len - len(trimmed[i].content)
        
        # Remove orphaned ToolMessages whose parent AIMessage was trimmed
        valid_tc_ids = set()
        for m in trimmed:
            if hasattr(m, 'tool_calls') and m.tool_calls:
                for tc in m.tool_calls:
                    valid_tc_ids.add(tc.get('id', ''))
        trimmed = [m for m in trimmed
                   if not isinstance(m, ToolMessage) or m.tool_call_id in valid_tc_ids]
        
        return trimmed

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.
        Includes retry with harder trimming if 413 token limit is hit.
        """
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        
        # Trim long tool outputs to stay within API token limits
        messages = self._trim_messages(messages)
        
        try:
            response = self.model.invoke(messages)
        except Exception as e:
            if "413" in str(e) or "tokens_limit_reached" in str(e):
                # Return graceful skip instead of destructive emergency retry
                # (stripping images makes the model answer blind → wrong answers)
                print("⚠️  413 token limit hit — skipping (context too large)")
                return {"messages": [AIMessage(content="[TOKEN_LIMIT_EXCEEDED] Unable to process — context too large for the model's input window.")]}
            else:
                raise
        
        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        return len(response.tool_calls) > 0

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls with error detection, cross-referencing, and conflict resolution.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []
        canonical_findings = []
        
        print(f"\n{'='*60}")
        print(f"🔧 Executing {len(tool_calls)} tool(s): {', '.join(c['name'] for c in tool_calls)}")
        print("="*60)

        for i, call in enumerate(tool_calls, 1):
            t_start = _time.time()
            print(f"\n[{i}/{len(tool_calls)}] 🛠️  {call['name']}")
            _args_short = {k: (str(v)[:60] + '...' if len(str(v)) > 60 else v) for k, v in call.get('args', {}).items()}
            print(f"  Args: {_args_short}")
            
            if call["name"] not in self.tools:
                print("  ❌ Invalid tool name")
                result = "invalid tool, please retry"
            else:
                try:
                    _invoke_args = self._resolve_image_paths(call.get("args", {}))
                    raw_result = self.tools[call["name"]].invoke(_invoke_args)
                    elapsed = _time.time() - t_start
                    
                    # Check if the tool returned an error dict
                    _is_error = False
                    if isinstance(raw_result, tuple) and len(raw_result) >= 1:
                        first = raw_result[0]
                        if isinstance(first, dict) and "error" in first:
                            _is_error = True
                            err_text = first['error'] or '(empty error)'
                            print(f"  ⚠️  Tool returned error ({elapsed:.1f}s): {err_text[:200]}")
                    
                    if not _is_error:
                        print(f"  ✅ Success ({elapsed:.1f}s)")
                    
                    # Normalize to canonical findings (skip errors)
                    if self.enable_conflict_resolution and not _is_error:
                        tool_type = self._get_tool_type(call["name"])
                        normalized = normalize_output(
                            raw_result, call["name"], tool_type,
                            **call.get("args", {})
                        )
                        canonical_findings.extend(normalized)
                        print(f"  📊 Normalized to {len(normalized)} finding(s)")

                        # --- Aggregate findings across turns by image key ---
                        try:
                            args = call.get("args", {}) or {}
                            image_key = None
                            if isinstance(args, dict):
                                if "image_path" in args:
                                    image_key = args.get("image_path")
                                elif "image_paths" in args:
                                    ips = args.get("image_paths")
                                    if isinstance(ips, (list, tuple)) and ips:
                                        image_key = ips[0]
                                    elif isinstance(ips, str):
                                        image_key = ips
                            # Fallback: try metadata from raw_result
                            if not image_key and isinstance(raw_result, tuple) and len(raw_result) > 1 and isinstance(raw_result[1], dict):
                                meta = raw_result[1]
                                if "image_path" in meta:
                                    image_key = meta.get("image_path")
                                elif "image_paths" in meta:
                                    mips = meta.get("image_paths")
                                    if isinstance(mips, (list, tuple)) and mips:
                                        image_key = mips[0]
                            if image_key:
                                ag = self.aggregated_findings.setdefault(image_key, [])
                                # append non-duplicate findings
                                for f in normalized:
                                    dup_key = (f.pathology, f.source_tool, getattr(f, "raw_value", None))
                                    if not any((ff.pathology, ff.source_tool, getattr(ff, "raw_value", None)) == dup_key for ff in ag):
                                        ag.append(f)

                                # Run conflict detection on aggregated set if multiple tools present
                                unique_tools = set(f.source_tool for f in ag)
                                if len(ag) > 1 and len(unique_tools) > 1:
                                    agg_conflicts = self.conflict_detector.detect_conflicts(ag)
                                    if agg_conflicts:
                                        agg_resolutions = []
                                        for conflict in agg_conflicts:
                                            relevant_findings = [f for f in ag if f.pathology == conflict.finding]
                                            # Fallback for GACL / non-standard conflicts
                                            if not relevant_findings and conflict.tools_involved:
                                                relevant_findings = [f for f in ag if f.source_tool in conflict.tools_involved]
                                            res = self.conflict_resolver.resolve_conflict(conflict, relevant_findings)
                                            agg_resolutions.append(res)

                                        # Append aggregated conflict summary to latest tool result
                                        if results:
                                            results[-1].content += "\n\n--- AGGREGATED CONFLICT ANALYSIS ---\n"
                                            n_def = sum(1 for r in agg_resolutions if r.get("should_defer", False))
                                            lines = [f"⚠️ {len(agg_conflicts)} aggregated conflict(s) detected across tools ({len(unique_tools)} tools)."]
                                            if n_def:
                                                lines.append(f"{n_def} flagged for human review.")
                                            for c, r in list(zip(agg_conflicts, agg_resolutions))[:3]:
                                                lines.append(f"  • {c.finding}: {c.severity} — {r.get('decision','N/A')} ({r.get('confidence',0):.0%})")
                                            results[-1].content += "\n".join(lines)

                                        # Save comprehensive aggregated log
                                        try:
                                            self._save_tool_calls_with_conflicts(results, ag, agg_conflicts, agg_resolutions)
                                        except Exception:
                                            pass
                        except Exception:
                            # Aggregation must never crash tool execution; log and continue
                            import traceback as _tb
                            _tb.print_exc()
                    
                    result = self._format_tool_result(raw_result, call["name"])
                except Exception as e:
                    elapsed = _time.time() - t_start
                    err_msg = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__}: {e!r}"
                    print(f"  ❌ Error ({elapsed:.1f}s): {err_msg[:200]}")
                    result = f"Error executing tool: {err_msg}"

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    args=call["args"],
                    content=str(result),
                )
            )
        
        # Cross-reference VQA ↔ classifier findings
        if self.enable_conflict_resolution and canonical_findings:
            self._cross_reference_findings(canonical_findings)
        
        # Conflict Detection (need findings from 2+ distinct tools)
        conflicts = []
        resolutions = []
        unique_tools = set(f.source_tool for f in canonical_findings)
        
        if self.enable_conflict_resolution and len(canonical_findings) > 1 and len(unique_tools) > 1:
            print(f"\n{'='*60}")
            print(f"🔍 CONFLICT DETECTION ({len(canonical_findings)} findings from {len(unique_tools)} tools)")
            print("="*60)
            
            conflicts = self.conflict_detector.detect_conflicts(canonical_findings)
            
            if conflicts:
                print(f"\n⚠️  Detected {len(conflicts)} conflict(s)!")
                print("\n" + "="*60)
                print("🔧 CONFLICT RESOLUTION")
                print("="*60)
                
                for ci, conflict in enumerate(conflicts, 1):
                    print(f"\n[{ci}/{len(conflicts)}] {conflict.to_summary()}")
                    
                    # Primary filter: exact pathology match
                    relevant_findings = [
                        f for f in canonical_findings 
                        if f.pathology == conflict.finding
                    ]
                    # Fallback for GACL / non-standard conflicts: use tools_involved
                    if not relevant_findings and conflict.tools_involved:
                        relevant_findings = [
                            f for f in canonical_findings
                            if f.source_tool in conflict.tools_involved
                        ]
                    
                    resolution = self.conflict_resolver.resolve_conflict(conflict, relevant_findings)
                    resolutions.append(resolution)
                    
                    print(f"  Resolution: {resolution['decision']}")
                    print(f"  Confidence: {resolution['confidence']:.1%}")
                    if resolution.get('should_defer', False):
                        print("  ⚠️  FLAGGED FOR HUMAN REVIEW")
                
                n_def = sum(1 for r in resolutions if r.get('should_defer', False))
                lines = [f"⚠️ {len(conflicts)} conflict(s) detected, {len(resolutions)} resolved."]
                if n_def:
                    lines.append(f"{n_def} flagged for human review.")
                for c, r in list(zip(conflicts, resolutions))[:3]:
                    lines.append(f"  • {c.finding}: {c.severity} — {r.get('decision','N/A')} ({r.get('confidence',0):.0%})")
                if len(conflicts) > 3:
                    lines.append(f"  … and {len(conflicts)-3} more (see logs)")
                if results:
                    results[-1].content += f"\n\n--- CONFLICT ANALYSIS ---\n" + "\n".join(lines)
            else:
                print("✅ No conflicts detected - all tools agree")
        elif self.enable_conflict_resolution and len(canonical_findings) > 0:
            print(f"\n📊 {len(canonical_findings)} finding(s) from {len(unique_tools)} tool(s) — need 2+ for conflict detection")
        
        # Save comprehensive logs
        if self.enable_conflict_resolution:
            self._save_tool_calls_with_conflicts(results, canonical_findings, conflicts, resolutions)
        else:
            self._save_tool_calls(results)
        
        print(f"\n{'='*60}")  
        print("✅ Tool execution complete")
        print("="*60 + "\n")

        return {"messages": results}
    
    @staticmethod
    def _format_tool_result(result, tool_name):
        """Convert raw tool output to compact, LLM-readable text."""
        if isinstance(result, tuple) and len(result) >= 1:
            first = result[0]
            if isinstance(first, dict) and "error" in first:
                return f"Error from {tool_name}: {first['error']}"
        if isinstance(result, str):
            return result[:800]
        if isinstance(result, tuple) and len(result) == 2:
            output, meta = result
            if isinstance(output, dict) and not any(k in output for k in ("response", "image_path", "segmentation_image_path")):
                numeric = {k: v for k, v in output.items() if isinstance(v, (int, float))}
                if numeric:
                    findings = {k: round(float(v), 3) for k, v in numeric.items() if float(v) > 0.05}
                    if findings:
                        lines = [f"  {k}: {v:.1%}" for k, v in sorted(findings.items(), key=lambda x: -x[1])]
                        return f"Findings (>5%):\n" + "\n".join(lines) + f"\n({len(numeric)-len(findings)} below 5%)"
                    return "No significant findings (all below 5%)."
            if isinstance(output, dict) and "response" in output:
                return f"Expert answer: {output['response']}"
            if isinstance(output, str):
                return output[:800]
            if isinstance(output, dict) and "segmentation_image_path" in output:
                metrics = output.get("metrics", {})
                parts = ["Segmentation results:"]
                for organ, m in list(metrics.items())[:6]:
                    if isinstance(m, dict):
                        parts.append(f"  {organ}: area={m.get('area_pixels','?')}px, conf={m.get('confidence_score','?'):.2f}")
                if len(metrics) > 6:
                    parts.append(f"  … and {len(metrics)-6} more organs")
                return "\n".join(parts)
            if isinstance(output, dict):
                if "image_path" in output:
                    return f"Output image: {output['image_path']}"
                return str(output)[:600]
        return str(result)[:600]

    def _cross_reference_findings(self, canonical_findings):
        """Align VQA text findings with classifier's pathology names for conflict detection."""
        clf_pathologies = set()
        vqa_info = {}
        existing_vqa = set()

        for f in canonical_findings:
            if f.evidence_type == "classification":
                clf_pathologies.add(f.pathology)
            elif f.evidence_type == "vqa":
                if f.pathology != "general_assessment":
                    existing_vqa.add((f.source_tool, f.pathology))
                raw = f.raw_value if isinstance(f.raw_value, str) else str(f.raw_value)
                vqa_info[f.source_tool] = (raw, f.confidence)

        if not clf_pathologies or not vqa_info:
            return

        new = []
        for tool, (raw_text, base_conf) in vqa_info.items():
            polarity = dict(extract_pathologies_with_polarity(raw_text))
            for path in clf_pathologies:
                if (tool, path) in existing_vqa:
                    continue
                if path in polarity:
                    # VQA explicitly mentions this pathology (positive or negative)
                    is_pos = polarity[path]
                    conf = base_conf if is_pos else max(0.05, 1.0 - base_conf)
                else:
                    # VQA never mentioned this pathology.
                    # Silence is NOT informative: CheXagent returns terse
                    # single-finding answers, so absence-of-mention does NOT
                    # mean the pathology is absent.  Creating a synthetic
                    # "absent at 20%" finding here leads to false CRITICAL
                    # conflicts with every classifier pathology at 60-71%.
                    continue
                new.append(CanonicalFinding(
                    pathology=path, region="unspecified",
                    confidence=conf, evidence_type="vqa",
                    source_tool=tool, raw_value=raw_text[:300],
                    metadata={"synthetic_cross_ref": True, "is_positive_mention": is_pos}
                ))
        canonical_findings.extend(new)
        if new:
            print(f"  🔗 Cross-referenced {len(new)} VQA findings to classifier pathologies")

    def _get_tool_type(self, tool_name: str) -> str:
        """Determine tool type from tool name."""
        if "classifier" in tool_name.lower() or "classification" in tool_name.lower():
            return "classification"
        elif "vqa" in tool_name.lower() or "expert" in tool_name.lower() or "llava" in tool_name.lower():
            return "vqa"
        elif "segmentation" in tool_name.lower():
            return "segmentation"
        elif "grounding" in tool_name.lower():
            return "grounding"
        elif "report" in tool_name.lower():
            return "report"
        else:
            return "unknown"

    @staticmethod
    def _resolve_image_paths(args):
        """Resolve relative image paths using MEDRAX_FIGURES_DIR env var.

        When the LLM passes a relative path like 'figures/11583/figure_1.jpg',
        this resolves it to the absolute path under BENCH_ROOT so tools can
        find the file on Kaggle or any deployment environment.
        """
        if not isinstance(args, dict):
            return args
        _fig_dir = os.environ.get("MEDRAX_FIGURES_DIR", "")
        if not _fig_dir:
            return args
        args = dict(args)  # shallow copy — don't mutate LangChain objects
        for key in ("image_path", "image_paths"):
            if key not in args:
                continue
            val = args[key]
            if isinstance(val, str) and val and not os.path.isabs(val) and not os.path.exists(val):
                candidate = os.path.join(_fig_dir, val)
                if os.path.exists(candidate):
                    args[key] = candidate
                    print(f"  \U0001f4c2 Resolved {key}: {val} \u2192 {candidate}")
            elif isinstance(val, (list, tuple)):
                new_vals = []
                changed = False
                for v in val:
                    if isinstance(v, str) and v and not os.path.isabs(v) and not os.path.exists(v):
                        c = os.path.join(_fig_dir, v)
                        if os.path.exists(c):
                            new_vals.append(c)
                            changed = True
                        else:
                            new_vals.append(v)
                    else:
                        new_vals.append(v)
                if changed:
                    args[key] = new_vals
                    print(f"  \U0001f4c2 Resolved {key} list ({len(new_vals)} items)")
        return args

    def _save_tool_calls_with_conflicts(
        self, 
        tool_calls: List[ToolMessage], 
        canonical_findings: List[CanonicalFinding],
        conflicts: List,
        resolutions: List[Dict[str, Any]]
    ) -> None:
        """
        Save comprehensive tool execution log including conflict analysis.

        Args:
            tool_calls: Raw tool messages
            canonical_findings: Normalized findings
            conflicts: Detected conflicts
            resolutions: Conflict resolutions
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive log
        comprehensive_log = {
            "timestamp": datetime.now().isoformat(),
            "session_id": timestamp,
            "summary": {
                "total_tools_called": len(tool_calls),
                "canonical_findings": len(canonical_findings),
                "conflicts_detected": len(conflicts),
                "conflicts_resolved": len(resolutions),
                "human_review_required": any(r.get('should_defer', False) for r in resolutions)
            },
            "tool_executions": [],
            "canonical_findings": [],
            "conflict_analysis": {
                "conflicts": [],
                "resolutions": []
            }
        }
        
        # Add tool execution details
        for call in tool_calls:
            comprehensive_log["tool_executions"].append({
                "tool_call_id": call.tool_call_id,
                "tool_name": call.name,
                "args": call.args,
                "result": call.content[:500] + "..." if len(str(call.content)) > 500 else call.content,
                "timestamp": datetime.now().isoformat(),
            })
        
        # Add canonical findings
        for finding in canonical_findings:
            comprehensive_log["canonical_findings"].append(finding.to_dict())
        
        # Add conflict details
        for i, conflict in enumerate(conflicts):
            comprehensive_log["conflict_analysis"]["conflicts"].append(conflict.to_dict())
            if i < len(resolutions):
                comprehensive_log["conflict_analysis"]["resolutions"].append(resolutions[i])
        
        # Save to file
        filename = self.log_path / f"conflict_resolution_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(comprehensive_log, f, indent=2)
        
        print(f"📝 Comprehensive log saved: {filename}")

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        """
        Save tool calls to a JSON file with timestamp-based naming.

        Args:
            tool_calls (List[ToolMessage]): List of tool calls to save.
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)
