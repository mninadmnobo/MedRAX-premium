import json
import operator
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional, Tuple
import copy

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from .canonical_output import normalize_output, CanonicalFinding
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
                # Emergency trim: strip ALL images, keep only last 6 messages
                print("⚠️  413 token limit hit — emergency trim and retry")
                emergency = []
                for m in messages:
                    if isinstance(m, SystemMessage):
                        emergency.append(m)
                    elif isinstance(m, HumanMessage):
                        emergency.append(self._strip_images_from_human(m))
                    elif isinstance(m, ToolMessage):
                        emergency.append(ToolMessage(
                            tool_call_id=m.tool_call_id,
                            name=m.name,
                            content=self._truncate_content(str(m.content), 150),
                        ))
                    elif isinstance(m, AIMessage):
                        new_m = copy.copy(m)
                        if isinstance(m.content, str):
                            new_m.content = self._truncate_content(m.content, 300)
                        emergency.append(new_m)
                    else:
                        emergency.append(m)
                # Keep system + last 6
                sys_msgs = [m for m in emergency if isinstance(m, SystemMessage)]
                rest = [m for m in emergency if not isinstance(m, SystemMessage)]
                messages = sys_msgs + rest[-6:]
                response = self.model.invoke(messages)
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
        Execute tool calls from the model's response with conflict detection and resolution.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []
        canonical_findings = []
        
        # STEP 1: Execute all tools
        print("\n" + "="*60)
        print(f"🔧 Executing {len(tool_calls)} tool(s)")
        print("="*60)

        for i, call in enumerate(tool_calls, 1):
            print(f"\n[{i}/{len(tool_calls)}] Tool: {call['name']}")
            
            if call["name"] not in self.tools:
                print("  ❌ Invalid tool")
                result = "invalid tool, please retry"
            else:
                try:
                    result = self.tools[call["name"]].invoke(call["args"])
                    print("  ✅ Execution successful")
                    
                    # STEP 2: Normalize output to canonical format (if conflict resolution enabled)
                    if self.enable_conflict_resolution:
                        tool_type = self._get_tool_type(call["name"])
                        normalized = normalize_output(
                            result, 
                            call["name"], 
                            tool_type,
                            **call.get("args", {})
                        )
                        canonical_findings.extend(normalized)
                        print(f"  📊 Normalized to {len(normalized)} finding(s)")
                except Exception as e:
                    print(f"  ❌ Error: {str(e)}")
                    result = f"Error executing tool: {str(e)}"

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    args=call["args"],
                    content=str(result),
                )
            )
        
        # STEP 3: Conflict Detection (if enabled and we have multiple findings)
        conflicts = []
        resolutions = []
        
        if self.enable_conflict_resolution and len(canonical_findings) > 1:
            print(f"\n{'='*60}")
            print("🔍 CONFLICT DETECTION")
            print("="*60)
            print(f"Analyzing {len(canonical_findings)} canonical findings...")
            
            conflicts = self.conflict_detector.detect_conflicts(canonical_findings)
            
            if conflicts:
                print(f"\n⚠️  Detected {len(conflicts)} conflict(s)!")
                
                # STEP 4: Resolve conflicts
                print("\n" + "="*60)
                print("🔧 CONFLICT RESOLUTION")
                print("="*60)
                
                for i, conflict in enumerate(conflicts, 1):
                    print(f"\n[{i}/{len(conflicts)}] {conflict.to_summary()}")
                    
                    # Get relevant findings for this conflict
                    relevant_findings = [
                        f for f in canonical_findings 
                        if f.pathology == conflict.finding
                    ]
                    
                    resolution = self.conflict_resolver.resolve_conflict(conflict, relevant_findings)
                    resolutions.append(resolution)
                    
                    print(f"  Resolution: {resolution['decision']}")
                    print(f"  Confidence: {resolution['confidence']:.1%}")
                    if resolution.get('should_defer', False):
                        print("  ⚠️  FLAGGED FOR HUMAN REVIEW")
                
                # Generate a COMPACT conflict summary for the tool message
                # (Full report is saved to JSON log; keep message small to avoid 413 token errors)
                n_deferred = sum(1 for r in resolutions if r.get('should_defer', False))
                summary_lines = [f"⚠️ {len(conflicts)} conflict(s) detected, {len(resolutions)} resolved."]
                if n_deferred:
                    summary_lines.append(f"{n_deferred} flagged for human review.")
                # Add top-3 most severe conflicts as one-liners
                for c, r in list(zip(conflicts, resolutions))[:3]:
                    summary_lines.append(f"  • {c.finding}: {c.severity} — {r.get('decision','N/A')} ({r.get('confidence',0):.0%})")
                if len(conflicts) > 3:
                    summary_lines.append(f"  … and {len(conflicts)-3} more (see logs)")
                compact_report = "\n".join(summary_lines)
                
                if results:
                    results[-1].content += f"\n\n--- CONFLICT ANALYSIS ---\n{compact_report}"
                print(f"\n📋 Conflict summary appended to last tool result")
            else:
                print("✅ No conflicts detected - all tools agree")
        
        # STEP 5: Save comprehensive logs
        if self.enable_conflict_resolution:
            self._save_tool_calls_with_conflicts(results, canonical_findings, conflicts, resolutions)
        else:
            self._save_tool_calls(results)
        
        print(f"\n{'='*60}")  
        print("✅ Tool execution complete")
        print("="*60 + "\n")
        print("Returning to model processing!")

        return {"messages": results}
    
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
