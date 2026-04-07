"""
Conflict detection and resolution for MedRAX tool outputs.
Implements explicit conflict detection with rule-based and probabilistic approaches.
Now includes BERT-based semantic conflict detection for textual outputs.

Premium Resolution: Argumentation Graph + Weighted Tool Trust + Uncertainty Abstention
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

from .canonical_output import CanonicalFinding
from .anatomical_consistency_graph import GACLConflictDetector
from .argumentation_graph import ArgumentGraphBuilder, ArgumentGraph
from .tool_trust import ToolTrustManager
from .abstention_logic import AbstentionLogic, AbstentionReason

# Note: BERT detector is now lazy-loaded within ConflictDetector class
# Use ConflictDetector.bert_detector property instead of a module-level function


@dataclass
class Conflict:
    """
    Represents a detected conflict between tool outputs.
    
    Attributes:
        conflict_type: Type of conflict ("presence", "location", "severity", "value", "semantic")
        finding: What finding is in conflict (e.g., "Pneumothorax")
        tools_involved: List of tool names that disagree
        values: Conflicting values from each tool
        confidences: Confidence scores from each tool
        severity: How critical is this conflict ("critical", "moderate", "minor")
        recommendation: Suggested resolution approach
        timestamp: When conflict was detected
        bert_scores: BERT-based conflict detection scores (for resolution pipeline)
    """
    conflict_type: str
    finding: str
    tools_involved: List[str]
    values: List[Any]
    confidences: List[float]
    severity: str
    recommendation: str
    timestamp: Optional[str] = None
    bert_scores: Optional[Dict[str, float]] = None  # NEW: BERT conflict scores
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.bert_scores is None:
            self.bert_scores = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_conflict_score(self) -> float:
        """Get the primary conflict score (BERT contradiction probability if available)."""
        if self.bert_scores and "contradiction_prob" in self.bert_scores:
            return self.bert_scores["contradiction_prob"]
        # Fallback: use confidence gap
        if len(self.confidences) >= 2:
            return max(self.confidences) - min(self.confidences)
        return 0.0
    
    def to_summary(self) -> str:
        """Generate human-readable summary."""
        values_str = " vs ".join([f"{t}={v} ({c:.0%})" 
                                   for t, v, c in zip(self.tools_involved, self.values, self.confidences)])
        score_str = ""
        if self.bert_scores and "contradiction_prob" in self.bert_scores:
            score_str = f" [BERT score: {self.bert_scores['contradiction_prob']:.1%}]"
        return f"[{self.severity.upper()}] {self.conflict_type} conflict on '{self.finding}': {values_str}{score_str}"


class ConflictDetector:
    """
    Detects conflicts in multi-tool outputs using BERT and rule-based methods.
    
    Primary detection method: BERT-based NLI for semantic conflict detection
    Fallback: Rule-based confidence gap analysis
    """
    
    # Conflict detection thresholds
    PRESENCE_THRESHOLD_HIGH = 0.7  # >70% = present
    PRESENCE_THRESHOLD_LOW = 0.3   # <30% = absent
    CONFIDENCE_GAP_THRESHOLD = 0.4  # Difference that triggers conflict
    
    # Standard pathology list
    PATHOLOGIES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
        "Emphysema", "Fibrosis", "Fracture", "Hernia", "Infiltration",
        "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
        "Support Devices"
    ]
    
    def __init__(
        self, 
        sensitivity: float = 0.4,
        use_bert: bool = True,
        bert_device: Optional[str] = None,
        bert_cache_dir: Optional[str] = None,
    ):
        """
        Initialize conflict detector.
        
        Args:
            sensitivity: Confidence gap threshold (0-1). Lower = more sensitive.
            use_bert: Whether to use BERT-based conflict detection (recommended)
            bert_device: Device for BERT model (cuda/cpu)
            bert_cache_dir: Cache directory for BERT model
        """
        self.sensitivity = sensitivity
        self.CONFIDENCE_GAP_THRESHOLD = sensitivity
        self.use_bert = use_bert
        self.bert_device = bert_device
        self.bert_cache_dir = bert_cache_dir
        
        # Initialize GACL for semantic conflict detection
        self.gacl_detector = GACLConflictDetector(anomaly_threshold=0.6)
    
    # Class-level BERT singleton — loaded once, shared across all instances.
    # Protected by _shared_bert_lock to prevent double-loading and by a
    # defensive hasattr check so gc.collect() / del agent can never clear it.
    _shared_bert_detector = None
    _shared_bert_failed = False

    @property
    def bert_detector(self):
        """Lazy load BERT detector on first use (shared across ALL instances).

        The singleton is stored on the *class* so ``del agent`` / ``gc.collect()``
        between benchmark cases cannot accidentally reclaim it.
        """
        # Fast-path: already loaded
        det = ConflictDetector._shared_bert_detector
        if det is not None:
            return det
        if ConflictDetector._shared_bert_failed or not self.use_bert:
            return None
        try:
            from .bert_conflict_detector import MedicalConflictDetector
            det = MedicalConflictDetector(
                device=self.bert_device,
                cache_dir=self.bert_cache_dir,
                conflict_threshold=0.6,
            )
            # Store on class — survives instance deletion
            ConflictDetector._shared_bert_detector = det
            print("✓ BERT conflict detector loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load BERT detector: {e}. Using rule-based detection.")
            ConflictDetector._shared_bert_failed = True
            self.use_bert = False
        return ConflictDetector._shared_bert_detector
    
    def detect_conflicts(self, findings: List[CanonicalFinding]) -> List[Conflict]:
        """
        Tool-level conflict detection.
        
        At most **1 conflict per pair of distinct tools**.
        If N tools are called → max C(N, 2) conflicts.
        
        Algorithm:
        1. Deduplicate findings: one entry per (tool, pathology), keep max confidence.
        2. For each unordered pair of tools, find the shared pathology with the
           worst present-vs-absent disagreement.
        3. If that gap exceeds the sensitivity threshold → 1 conflict.
        
        Args:
            findings: List of canonical findings from all tools
            
        Returns:
            List of detected conflicts (≤ C(N,2) where N = number of distinct tools)
        """
        # --- Step 1: deduplicate ------------------------------------------------
        # When the same classifier runs on 3 images we get 3×18 = 54 findings.
        # Keep only the highest-confidence entry per (tool, pathology).
        by_tool: Dict[str, Dict[str, CanonicalFinding]] = {}
        for f in findings:
            tool = f.source_tool
            path = f.pathology
            if tool not in by_tool:
                by_tool[tool] = {}
            existing = by_tool[tool].get(path)
            if existing is None or f.confidence > existing.confidence:
                by_tool[tool][path] = f

        tool_names = list(by_tool.keys())
        if len(tool_names) < 2:
            return []

        # --- Step 2: one comparison per tool pair --------------------------------
        conflicts: List[Conflict] = []
        for i in range(len(tool_names)):
            for j in range(i + 1, len(tool_names)):
                conflict = self._compare_tool_pair(
                    tool_names[i], by_tool[tool_names[i]],
                    tool_names[j], by_tool[tool_names[j]],
                )
                if conflict:
                    conflicts.append(conflict)

        # --- Step 3: GACL anatomical consistency check ----------------------------
        # If segmentation + classification findings coexist, GACL can detect
        # anatomical inconsistencies (e.g. abnormal measurements vs normal class)
        # that rule-based confidence-gap checks would miss.
        gacl_conflict = self._run_gacl_check(by_tool)
        if gacl_conflict:
            # Only add if not already covered by a pairwise conflict on same tools
            already_covered = any(
                set(c.tools_involved) == set(gacl_conflict.tools_involved)
                and c.finding == gacl_conflict.finding
                for c in conflicts
            )
            if not already_covered:
                conflicts.append(gacl_conflict)

        return conflicts

    # ------------------------------------------------------------------
    # Helper: GACL anatomical consistency check
    # ------------------------------------------------------------------
    def _run_gacl_check(
        self, by_tool: Dict[str, Dict[str, "CanonicalFinding"]]
    ) -> Optional["Conflict"]:
        """Run GACL if both segmentation and classification tools are present.

        GACL compares anatomical measurements from segmentation against the
        classifier's prediction.  It can catch subtle structural anomalies
        (e.g. early cardiomyopathy) that a pure confidence-gap rule misses.
        Returns a single Conflict or None.
        """
        seg_tool = clf_tool = None
        for tool_name, findings_map in by_tool.items():
            sample = next(iter(findings_map.values()), None)
            if sample is None:
                continue
            if sample.evidence_type == "segmentation":
                seg_tool = tool_name
            elif sample.evidence_type == "classification":
                clf_tool = tool_name

        if not seg_tool or not clf_tool:
            return None

        try:
            # Build segmentation output dict from raw values
            seg_output: Dict[str, Any] = {}
            for _path, finding in by_tool[seg_tool].items():
                if isinstance(finding.raw_value, dict):
                    seg_output.update(finding.raw_value)
                else:
                    seg_output[finding.region] = {
                        "confidence_score": finding.confidence
                    }

            # Build classification output (highest-confidence pathology)
            clf_findings = by_tool[clf_tool]
            best_clf = max(clf_findings.values(), key=lambda f: f.confidence)
            is_normal = best_clf.confidence < 0.5 or best_clf.pathology in (
                "No Finding", "unknown"
            )
            clf_output = {
                "class": "Normal" if is_normal else best_clf.pathology,
                "score": best_clf.confidence,
            }

            result = self.gacl_detector.detect_semantic_conflict(
                seg_output, clf_output
            )

            if result.get("has_conflict"):
                anomaly = result.get("anomaly_score", 0.5)
                sev = "critical" if anomaly > 0.8 else (
                    "moderate" if anomaly > 0.6 else "minor"
                )
                return Conflict(
                    conflict_type="anatomical_consistency",
                    finding=result.get(
                        "most_likely_diagnosis", "anatomical pattern"
                    ),
                    tools_involved=[seg_tool, clf_tool],
                    values=[
                        result["segmentation_pattern"],
                        result["classification_prediction"],
                    ],
                    confidences=[
                        result.get("confidence_in_diagnosis", 0.5),
                        result.get("classification_confidence", 0.5),
                    ],
                    severity=sev,
                    recommendation=result.get(
                        "explanation",
                        "Review anatomical measurements vs classification.",
                    ),
                    bert_scores={
                        "gacl_anomaly_score": anomaly,
                    },
                )
        except Exception as e:
            print(f"  \u26a0\ufe0f GACL check failed: {e}")

        return None

    # ------------------------------------------------------------------
    # Helper: effective confidence (fixes VQA cross-ref calibration)
    # ------------------------------------------------------------------
    def _effective_confidence(self, f: CanonicalFinding) -> float:
        """Return calibrated confidence for presence/absence comparison.

        Synthetic VQA cross-refs carry the VQA's raw (uncalibrated) base
        confidence which is typically very low (~0.20).  That does NOT mean
        the VQA thinks the pathology is absent.  Fix:
        - Positive mention  → treat as *present*  (≥ 0.55)
        - Negative mention  → treat as *absent*   (≤ 0.20)
        - Non-synthetic     → use original confidence as-is
        """
        meta = f.metadata or {}
        if meta.get("synthetic_cross_ref"):
            if meta.get("is_positive_mention") is True:
                return max(f.confidence, 0.55)
            if meta.get("is_positive_mention") is False:
                return min(f.confidence, 0.20)
        return f.confidence

    # ------------------------------------------------------------------
    # Helper: compare one pair of tools → 0 or 1 conflict
    # ------------------------------------------------------------------
    def _compare_tool_pair(
        self,
        tool1: str, findings1: Dict[str, CanonicalFinding],
        tool2: str, findings2: Dict[str, CanonicalFinding],
    ) -> Optional[Conflict]:
        """Compare two tools on shared pathologies; return worst disagreement.

        Only pathologies where **both** tools have an opinion are checked.
        Meta-entries like ``general_assessment`` are skipped.
        """
        skip = {"general_assessment", "unknown", "No Finding"}
        p1 = {k: v for k, v in findings1.items() if k not in skip}
        p2 = {k: v for k, v in findings2.items() if k not in skip}
        shared = set(p1) & set(p2)
        if not shared:
            return None

        worst_path: Optional[str] = None
        worst_gap = 0.0

        for path in shared:
            c1 = self._effective_confidence(p1[path])
            c2 = self._effective_confidence(p2[path])
            present1 = c1 > 0.5
            present2 = c2 > 0.5

            if present1 != present2:
                gap = abs(c1 - c2)
                if gap > worst_gap:
                    worst_gap = gap
                    worst_path = path

        if not worst_path or worst_gap < self.CONFIDENCE_GAP_THRESHOLD:
            return None

        f1, f2 = p1[worst_path], p2[worst_path]
        c1 = self._effective_confidence(f1)
        c2 = self._effective_confidence(f2)
        severity = "critical" if worst_gap > 0.5 else ("moderate" if worst_gap > 0.3 else "minor")

        # --- Attach BERT-NLI scores if available (enriches resolution) ---
        bert_scores = self._run_bert_on_pair(f1, f2)
        if bert_scores:
            # Let BERT override severity when it has a strong opinion
            cp = bert_scores.get("contradiction_prob", 0.0)
            if cp > 0.85:
                severity = "critical"
            elif cp > 0.70 and severity == "minor":
                severity = "moderate"

        return Conflict(
            conflict_type="presence",
            finding=worst_path,
            tools_involved=[tool1, tool2],
            values=[
                f"present ({c1:.0%})" if c1 > 0.5 else f"absent ({c1:.0%})",
                f"present ({c2:.0%})" if c2 > 0.5 else f"absent ({c2:.0%})",
            ],
            confidences=[c1, c2],
            severity=severity,
            recommendation=self._get_presence_resolution_strategy([f1, f2]),
            bert_scores=bert_scores,
        )
    
    # ------------------------------------------------------------------
    # Helper: run BERT-NLI on a pair of findings (enriches the Conflict)
    # ------------------------------------------------------------------
    def _run_bert_on_pair(
        self,
        f1: CanonicalFinding,
        f2: CanonicalFinding,
    ) -> Optional[Dict[str, Any]]:
        """Run BERT-NLI on two findings and return score dict (or None).

        This enriches the Conflict object with semantic analysis without
        creating additional conflicts.  If BERT is unavailable or fails
        gracefully, the conflict still proceeds with rule-based scores.
        """
        if not self.use_bert:
            return None
        try:
            det = self.bert_detector
            if det is None:
                return None
            text1 = self._extract_text_from_finding(f1)
            text2 = self._extract_text_from_finding(f2)
            if not text1 or not text2:
                return None
            prediction = det.detect_conflict(
                text1=text1, text2=text2,
                tool1_name=f1.source_tool, tool2_name=f2.source_tool,
            )
            return {
                "contradiction_prob": prediction.conflict_probability,
                "entailment_prob": prediction.entailment_prob,
                "neutral_prob": prediction.neutral_prob,
                "conflict_type": prediction.conflict_type,
                "threshold_used": det.conflict_threshold,
                "text1_preview": text1[:200],
                "text2_preview": text2[:200],
            }
        except Exception as e:
            print(f"  ⚠️ BERT enrichment failed: {e}")
            return None
    
    def _get_presence_resolution_strategy(self, findings: List[CanonicalFinding]) -> str:
        """Recommend how to resolve presence/absence conflict."""
        # Find highest confidence finding
        max_finding = max(findings, key=lambda f: f.confidence)
        
        if max_finding.confidence > 0.85:
            return f"High confidence from {max_finding.source_tool} ({max_finding.confidence:.0%}). Recommend: Trust primary tool, but flag for review."
        elif max_finding.confidence > 0.7:
            return f"Moderate confidence from {max_finding.source_tool} ({max_finding.confidence:.0%}). Recommend: Call additional verification tool."
        else:
            return "Low confidence across all tools. Recommend: Defer to radiologist review."
    
    def _extract_text_from_finding(self, finding: CanonicalFinding) -> str:
        """
        Extract textual description from a finding.
        
        Args:
            finding: CanonicalFinding object
            
        Returns:
            Text representation of the finding
        """
        # Try to get text from various sources
        if finding.metadata:
            # Check metadata for text fields
            for key in ["text", "description", "report", "findings", "output"]:
                if key in finding.metadata and finding.metadata[key]:
                    return str(finding.metadata[key])
        
        # Use raw_value if it's text
        if isinstance(finding.raw_value, str):
            return finding.raw_value
        elif isinstance(finding.raw_value, dict):
            # Try common keys
            for key in ["text", "description", "report", "findings", "output", "prediction"]:
                if key in finding.raw_value:
                    return str(finding.raw_value[key])
            # Fallback to string representation
            return str(finding.raw_value)
        
        # Construct from finding attributes
        presence = "present" if finding.confidence > 0.5 else "absent"
        return f"{finding.pathology} is {presence} in {finding.region} with {finding.confidence:.0%} confidence"


class ConflictResolver:
    """
    Resolves conflicts using PREMIUM strategy: Argumentation Graph + Weighted Tool Trust + Uncertainty Abstention
    
    Resolution Strategy (Premium - with three new components):
    1. Build ArgumentGraph: Structure disagreement as explicit support/attack positions
    2. Apply ToolTrust: Weight opinions by historical tool reliability (learns over time)
    3. Check Abstention: Know when to say "I don't know, needs human review"
    4. Fallback to original logic: BERT scores + task-aware arbitration (backward compatible)
    
    Backward Compatibility:
    - Returns same dict format as before
    - Adds new optional fields: argumentation_graph, tool_weights_used, abstention_reason
    - Existing tests continue to pass
    - Original logic still applies as fallback
    """
    
    # Tool expertise hierarchy: which tool is best for which task
    TOOL_EXPERTISE = {
        "presence_detection": {
            "primary": "chest_xray_classifier",
            "fallback": "chest_xray_expert",
            "description": "Binary presence/absence of pathology"
        },
        "localization": {
            "primary": "chest_xray_segmentation",
            "fallback": "xray_phrase_grounding",
            "description": "Precise anatomical location"
        },
        "description": {
            "primary": "chest_xray_expert",
            "fallback": "chest_xray_report_generator",
            "description": "Detailed clinical description"
        },
        "severity_assessment": {
            "primary": "chest_xray_expert",
            "fallback": "chest_xray_classifier",
            "description": "Severity grading (mild/moderate/severe)"
        }
    }
    
    # BERT score thresholds for resolution decisions
    BERT_HIGH_CONTRADICTION = 0.85  # Very confident contradiction
    BERT_MODERATE_CONTRADICTION = 0.70  # Moderate contradiction
    BERT_HIGH_ENTAILMENT = 0.70  # Tools agree (not a real conflict)
    
    def __init__(
        self, 
        deferral_threshold: float = 0.6,
        enable_argumentation: bool = True,
        enable_tool_trust: bool = True,
        enable_abstention: bool = True,
        trust_weights_file: Optional[str] = None
    ):
        """
        Initialize conflict resolver with premium features.
        
        Args:
            deferral_threshold: Confidence below which we defer to humans
            enable_argumentation: Use argumentation graphs? (default: True)
            enable_tool_trust: Use learned tool trust weights? (default: True)
            enable_abstention: Use abstention logic? (default: True)
            trust_weights_file: JSON file to persist/load tool trust weights
                               If None, will use default location in agent folder
        """
        self.deferral_threshold = deferral_threshold
        self.enable_argumentation = enable_argumentation
        self.enable_tool_trust = enable_tool_trust
        self.enable_abstention = enable_abstention
        
        # Initialize premium components
        self.argument_builder = ArgumentGraphBuilder()
        
        # Setup trust weights file
        if trust_weights_file is None:
            # Default location: same folder as this file
            trust_weights_file = os.path.join(
                os.path.dirname(__file__),
                "tool_trust_weights.json"
            )
        self.trust_manager = ToolTrustManager(persistence_file=trust_weights_file)
        
        # Initialize abstention logic
        self.abstention_logic = AbstentionLogic()
    
    def resolve_conflict(self, conflict: Conflict, findings: List[CanonicalFinding]) -> Dict[str, Any]:
        """
        Resolve a conflict using PREMIUM strategy + fallback logic.
        
        NEW Resolution Pipeline (PREMIUM):
        1. Build ArgumentGraph: Visualize support/attack structure
        2. Apply ToolTrust: Weight by learned reliability
        3. Check Abstention: Know when to say "I don't know"
        
        FALLBACK (Original logic - backward compatible):
        4. BERT scores: High entailment → not a real conflict
        5. BERT-guided: High contradiction + confidence leader
        6. Task-aware arbitration: Trust primary expert tool
        7. Weighted average: Last resort
        
        Args:
            conflict: Detected conflict (may contain bert_scores)
            findings: All findings related to this conflict
            
        Returns:
            Resolution dict with decision, confidence, reasoning
            NEW fields: argumentation_graph, tool_weights_used, abstention_reason (when applicable)
        """
        resolution = {}
        
        # STEP 1: Analyze BERT scores if available
        bert_analysis = self._analyze_bert_scores(conflict)
        resolution["bert_analysis"] = bert_analysis
        
        # If BERT indicates high entailment, this may not be a real conflict
        if bert_analysis.get("is_false_positive"):
            return {
                "decision": "bert_entailment_detected",
                "confidence": bert_analysis["entailment_prob"],
                "reasoning": f"BERT indicates agreement ({bert_analysis['entailment_prob']:.0%} entailment). "
                             f"This may be a false positive conflict.",
                "should_defer": False,
                "bert_analysis": bert_analysis,
                "value": None,
            }
        
        # ===== COMPUTE TRUST WEIGHTS (independent of argumentation) =====
        trust_weights = None
        if self.enable_tool_trust:
            trust_weights = {
                finding.source_tool: self.trust_manager.get_weight(finding.source_tool)
                for finding in findings
            }

        # ===== PREMIUM COMPONENT #1: BUILD ARGUMENT GRAPH =====
        argument_graph = None
        if self.enable_argumentation and len(findings) >= 1:
            try:
                
                # Build the argument graph
                argument_graph = self.argument_builder.build_from_conflict(
                    claim=f"{conflict.finding} present",
                    tools_involved=[f.source_tool for f in findings],
                    confidences=[f.confidence for f in findings],
                    values=[f.pathology for f in findings],
                    tool_trust_weights=trust_weights
                )
                resolution["argumentation_graph"] = argument_graph.to_dict()
                
                if self.enable_tool_trust and trust_weights:
                    resolution["tool_weights_used"] = trust_weights
            except Exception as e:
                print(f"⚠️  Argumentation graph building failed: {e}")
        
        # ===== PREMIUM COMPONENT #2: CHECK ABSTENTION =====
        abstention_reason = None
        if self.enable_abstention and argument_graph:
            try:
                abstention_decision = self.abstention_logic.should_abstain(
                    support_strength=argument_graph.support_strength,
                    attack_strength=argument_graph.attack_strength,
                    certainty=argument_graph.certainty,
                    has_cycles=argument_graph.has_cycles,
                    clinical_severity=conflict.severity,
                    num_tools=len(findings),
                    bert_contradiction_prob=bert_analysis.get("contradiction_prob", 0.0)
                )
                
                if abstention_decision.should_abstain:
                    abstention_reason = abstention_decision.reason.value
                    resolution["abstention_reason"] = abstention_reason
                    resolution["abstention_explanation"] = abstention_decision.explanation
                    resolution["risk_level"] = abstention_decision.risk_level
                    
                    return {
                        "decision": "abstained",
                        "confidence": 0.0,
                        "value": None,
                        "reasoning": f"Abstaining from resolution: {abstention_decision.explanation}",
                        "should_defer": True,
                        "abstention_reason": abstention_reason,
                        "abstention_explanation": abstention_decision.explanation,
                        "risk_level": abstention_decision.risk_level,
                        "argumentation_graph": argument_graph.to_dict(),
                        "bert_analysis": bert_analysis,
                    }
            except Exception as e:
                print(f"⚠️  Abstention check failed: {e}")
        
        # ===== PREMIUM COMPONENT #3: ARGUMENT GRAPH + TOOL TRUST RESOLUTION =====
        # If the graph has a clear winner, use it to resolve — this is the
        # main value-add of the premium pipeline.  Only fall through to the
        # legacy fallback when the graph is inconclusive ("unclear").
        if argument_graph and argument_graph.net_winner != "unclear" and argument_graph.certainty > 0.55:
            is_present = argument_graph.net_winner == "support"
            winning_nodes = argument_graph.support_nodes if is_present else argument_graph.attack_nodes
            if winning_nodes:
                best_node = max(winning_nodes, key=lambda n: n.strength)
                # Confidence = graph certainty adjusted by BERT severity
                adj_conf = argument_graph.certainty * bert_analysis.get("severity_adjustment", 1.0)
                return {
                    "decision": "argumentation_graph_winner",
                    "selected_tool": best_node.tool_name,
                    "value": is_present,
                    "confidence": adj_conf,
                    "reasoning": (
                        f"Argumentation graph resolves conflict: {argument_graph.net_winner} wins "
                        f"(certainty={argument_graph.certainty:.0%}, "
                        f"gap={argument_graph.confidence_gap:.2f}). "
                        f"Strongest contributor: {best_node.tool_name} "
                        f"({best_node.confidence:.0%}, trust-weighted strength: {best_node.strength:.2f})."
                    ),
                    "should_defer": adj_conf < self.deferral_threshold,
                    "argumentation_graph": argument_graph.to_dict(),
                    "tool_weights_used": trust_weights or {},
                    "bert_analysis": bert_analysis,
                }
        
        # ===== FALLBACK: ORIGINAL RESOLUTION LOGIC (Backward Compatible) =====
        
        # STEP 2: BERT-guided resolution for high-confidence contradictions
        if bert_analysis.get("contradiction_prob", 0) > self.BERT_HIGH_CONTRADICTION:
            bert_resolution = self._bert_guided_resolution(findings, bert_analysis)
            if bert_resolution:
                # Add premium components if available
                if argument_graph:
                    bert_resolution["argumentation_graph"] = argument_graph.to_dict()
                if resolution.get("tool_weights_used"):
                    bert_resolution["tool_weights_used"] = resolution["tool_weights_used"]
                return bert_resolution
        
        # STEP 3: Determine task type and use expertise hierarchy
        task_type = self._get_task_type(conflict.conflict_type)
        expertise = self.TOOL_EXPERTISE.get(task_type, {})
        primary_tool = expertise.get("primary")
        fallback_tool = expertise.get("fallback")
        
        # Find primary tool's finding
        primary_finding = self._find_tool_finding(findings, primary_tool)
        fallback_finding = self._find_tool_finding(findings, fallback_tool)
        
        # STEP 4: Trust primary tool if available and confident enough
        if primary_finding:
            # Adjust trust based on BERT contradiction level
            adjusted_confidence = self._adjust_confidence_by_bert(
                primary_finding.confidence, 
                bert_analysis
            )
            # Factor in learned tool trust weight
            if trust_weights:
                tw = trust_weights.get(primary_tool, 1.0)
                adjusted_confidence *= tw
            
            resolution = {
                "decision": "trust_primary_tool",
                "selected_tool": primary_tool,
                "value": primary_finding.confidence > 0.5,
                "confidence": adjusted_confidence,
                "reasoning": (
                    f"Primary tool for {task_type} is {primary_tool} "
                    f"(trust={trust_weights.get(primary_tool, 1.0):.0%}). "
                    f"BERT contradiction: {bert_analysis.get('contradiction_prob', 0):.0%}"
                ) if trust_weights else (
                    f"Primary tool for {task_type} is {primary_tool}. "
                    f"BERT contradiction: {bert_analysis.get('contradiction_prob', 0):.0%}"
                ),
                "should_defer": adjusted_confidence < self.deferral_threshold,
                "bert_analysis": bert_analysis,
            }
        elif fallback_finding:
            # Use fallback tool
            adjusted_confidence = self._adjust_confidence_by_bert(
                fallback_finding.confidence,
                bert_analysis
            )
            if trust_weights:
                tw = trust_weights.get(fallback_tool, 1.0)
                adjusted_confidence *= tw
            
            resolution = {
                "decision": "trust_fallback_tool",
                "selected_tool": fallback_tool,
                "value": fallback_finding.confidence > 0.5,
                "confidence": adjusted_confidence,
                "reasoning": (
                    f"Using fallback tool {fallback_tool} for {task_type} "
                    f"(trust={trust_weights.get(fallback_tool, 1.0):.0%}). "
                    f"BERT contradiction: {bert_analysis.get('contradiction_prob', 0):.0%}"
                ) if trust_weights else (
                    f"Using fallback tool {fallback_tool} for {task_type}. "
                    f"BERT contradiction: {bert_analysis.get('contradiction_prob', 0):.0%}"
                ),
                "should_defer": adjusted_confidence < self.deferral_threshold,
                "bert_analysis": bert_analysis,
            }
        else:
            # STEP 5: Weighted average resolution as last resort
            resolution = self._weighted_average_resolution(findings, bert_analysis)
        
        # Add premium components if available
        if argument_graph and "argumentation_graph" not in resolution:
            resolution["argumentation_graph"] = argument_graph.to_dict()
        if resolution.get("tool_weights_used") is None and self.enable_tool_trust:
            resolution["tool_weights_used"] = {
                f.source_tool: self.trust_manager.get_weight(f.source_tool)
                for f in findings
            }
        
        return resolution
    
    def _analyze_bert_scores(self, conflict: Conflict) -> Dict[str, Any]:
        """
        Analyze BERT scores from conflict to guide resolution.
        
        Returns:
            Dict with analysis results:
            - contradiction_prob: How certain BERT is of contradiction
            - entailment_prob: How certain BERT is of agreement
            - neutral_prob: How certain BERT is of no relation
            - is_false_positive: True if high entailment suggests not a real conflict
            - severity_adjustment: Multiplier for confidence adjustment
        """
        bert_scores = conflict.bert_scores or {}
        
        contradiction = bert_scores.get("contradiction_prob", 0.0)
        entailment = bert_scores.get("entailment_prob", 0.0)
        neutral = bert_scores.get("neutral_prob", 0.0)
        
        # Check for false positive (high entailment = tools actually agree)
        is_false_positive = entailment > self.BERT_HIGH_ENTAILMENT and contradiction < 0.3
        
        # Calculate severity adjustment based on contradiction confidence.
        # When no BERT scores are available (tool-level detection without
        # BERT), severity_adjustment = 1.0 so we don't penalise confidence.
        if not bert_scores:
            severity_adjustment = 1.0  # No BERT involved, no discount
        elif contradiction > self.BERT_HIGH_CONTRADICTION:
            severity_adjustment = 1.0  # High severity, no discount
        elif contradiction > self.BERT_MODERATE_CONTRADICTION:
            severity_adjustment = 0.9  # Slight confidence reduction
        else:
            severity_adjustment = 0.8  # More discount for uncertain conflicts
        
        return {
            "contradiction_prob": contradiction,
            "entailment_prob": entailment,
            "neutral_prob": neutral,
            "is_false_positive": is_false_positive,
            "severity_adjustment": severity_adjustment,
            "text1_preview": bert_scores.get("text1_preview", ""),
            "text2_preview": bert_scores.get("text2_preview", ""),
        }
    
    def _bert_guided_resolution(
        self, 
        findings: List[CanonicalFinding],
        bert_analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Use BERT contradiction confidence to guide resolution.
        
        When BERT is highly confident about contradiction:
        - Trust the finding with higher original confidence
        - Flag for human review if both have similar confidence
        
        Returns:
            Resolution dict or None if BERT guidance doesn't apply
        """
        if len(findings) < 2:
            return None
        
        # Sort by confidence descending
        sorted_findings = sorted(findings, key=lambda f: f.confidence, reverse=True)
        highest = sorted_findings[0]
        second = sorted_findings[1]
        
        confidence_gap = highest.confidence - second.confidence
        
        # If clear winner by confidence, trust it
        if confidence_gap > 0.3:
            return {
                "decision": "bert_high_confidence_leader",
                "selected_tool": highest.source_tool,
                "value": highest.confidence > 0.5,
                "confidence": highest.confidence * bert_analysis["severity_adjustment"],
                "reasoning": (
                    f"BERT detected high contradiction ({bert_analysis['contradiction_prob']:.0%}). "
                    f"Trusting {highest.source_tool} with significantly higher confidence "
                    f"({highest.confidence:.0%} vs {second.confidence:.0%})."
                ),
                "should_defer": False,
                "bert_analysis": bert_analysis,
            }
        
        # Close confidence scores + high BERT contradiction = defer to human
        if confidence_gap < 0.15 and bert_analysis["contradiction_prob"] > 0.8:
            # Use the higher tool's confidence (discounted) so downstream
            # analysis still has a meaningful number instead of 0.
            adj_conf = highest.confidence * bert_analysis.get("severity_adjustment", 0.8) * 0.5
            return {
                "decision": "bert_requires_human_review",
                "selected_tool": None,
                "value": None,
                "confidence": adj_conf,
                "reasoning": (
                    f"BERT detected strong contradiction ({bert_analysis['contradiction_prob']:.0%}), "
                    f"but tool confidences are too close ({highest.confidence:.0%} vs {second.confidence:.0%}). "
                    f"Deferring to radiologist review."
                ),
                "should_defer": adj_conf < self.deferral_threshold,
                "bert_analysis": bert_analysis,
            }
        
        # Moderate gap (0.15–0.30): trust the higher-confidence tool but flag for review
        if confidence_gap >= 0.15:
            adjusted = highest.confidence * bert_analysis["severity_adjustment"]
            return {
                "decision": "trust_primary_tool",
                "selected_tool": highest.source_tool,
                "value": highest.confidence > 0.5,
                "confidence": adjusted,
                "reasoning": (
                    f"BERT detected contradiction ({bert_analysis['contradiction_prob']:.0%}). "
                    f"Moderate confidence gap — trusting {highest.source_tool} "
                    f"({highest.confidence:.0%} vs {second.confidence:.0%})."
                ),
                "should_defer": adjusted < self.deferral_threshold,
                "bert_analysis": bert_analysis,
            }
        
        return None
    
    def _get_task_type(self, conflict_type: str) -> str:
        """Map conflict type to task type for expertise lookup."""
        mapping = {
            "presence": "presence_detection",
            "location": "localization",
            "severity": "severity_assessment",
            "semantic": "description",
            "value": "presence_detection",
            "anatomical_consistency": "presence_detection",
        }
        return mapping.get(conflict_type, "description")
    
    def _find_tool_finding(
        self, 
        findings: List[CanonicalFinding], 
        tool_name: Optional[str]
    ) -> Optional[CanonicalFinding]:
        """Find finding from a specific tool."""
        if not tool_name:
            return None
        for finding in findings:
            if finding.source_tool == tool_name:
                return finding
        return None
    
    def _adjust_confidence_by_bert(
        self, 
        confidence: float, 
        bert_analysis: Dict[str, Any]
    ) -> float:
        """
        Adjust tool confidence based on BERT analysis.
        
        If BERT is very confident about contradiction, we trust tool confidence as-is.
        If BERT is uncertain, we reduce the tool's effective confidence.
        """
        return confidence * bert_analysis.get("severity_adjustment", 1.0)
    
    def _weighted_average_resolution(
        self, 
        findings: List[CanonicalFinding],
        bert_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve conflict using trust-weighted confidence average.
        
        Each finding's influence = confidence * trust_weight, so historically
        reliable tools contribute more to the final decision.
        
        Args:
            findings: All findings for this pathology
            bert_analysis: Optional BERT analysis for additional context
            
        Returns:
            Resolution dict
        """
        if not findings:
            return {
                "decision": "insufficient_confidence",
                "confidence": 0.0,
                "value": None,
                "reasoning": "No findings available for resolution",
                "should_defer": True,
                "bert_analysis": bert_analysis or {},
            }

        # Compute per-finding weight = confidence * trust
        weights = []
        for f in findings:
            tw = self.trust_manager.get_weight(f.source_tool) if self.enable_tool_trust else 1.0
            weights.append(f.confidence * tw)

        total_weight = sum(weights)

        if total_weight == 0:
            return {
                "decision": "insufficient_confidence",
                "confidence": 0.0,
                "value": None,
                "reasoning": "All tools have zero confidence",
                "should_defer": True,
                "bert_analysis": bert_analysis or {},
            }

        # Trust-weighted presence vote
        weighted_presence = sum(
            w * (1.0 if f.confidence > 0.5 else 0.0)
            for f, w in zip(findings, weights)
        ) / total_weight

        # Trust-weighted average confidence
        avg_confidence = sum(weights) / sum(
            (self.trust_manager.get_weight(f.source_tool) if self.enable_tool_trust else 1.0)
            for f in findings
        )

        # Apply BERT severity adjustment
        if bert_analysis:
            avg_confidence *= bert_analysis.get("severity_adjustment", 1.0)

        return {
            "decision": "weighted_average",
            "value": weighted_presence > 0.5,
            "confidence": avg_confidence,
            "reasoning": (
                f"Trust-weighted average of {len(findings)} tool outputs. "
                f"Presence vote: {weighted_presence:.0%}"
            ),
            "should_defer": avg_confidence < self.deferral_threshold,
            "bert_analysis": bert_analysis or {},
        }
    
    def should_defer_to_human(self, resolution: Dict[str, Any]) -> bool:
        """
        Decide if this case should be deferred to human review.
        
        Args:
            resolution: Resolution decision
            
        Returns:
            True if should defer to radiologist
        """
        return resolution.get("should_defer", False)
    
    def update_trust_from_resolution(
        self,
        resolution: Dict[str, Any],
        was_correct: bool,
        findings: List[CanonicalFinding]
    ) -> Dict[str, float]:
        """
        Update tool trust weights based on resolution feedback.
        
        Call this after a resolution is confirmed correct or incorrect by radiologist.
        
        Args:
            resolution: The resolution dict returned by resolve_conflict()
            was_correct: True if resolution was confirmed correct, False if wrong
            findings: Original findings involved in the conflict
            
        Returns:
            Updated trust weights for all tools involved
        """
        if not self.enable_tool_trust:
            return {}
        
        # Update the selected tool (if one was chosen)
        selected_tool = resolution.get("selected_tool")
        if selected_tool:
            self.trust_manager.update_trust(selected_tool, was_correct)
        
        # Also update other tools based on whether their prediction aligned with outcome
        for finding in findings:
            tool_name = finding.source_tool
            # Tool was "correct" if its presence/absence aligned with the resolution
            tool_was_correct = (finding.confidence > 0.5) == resolution.get("value", False)
            if resolution.get("decision") != "abstained":
                self.trust_manager.update_trust(tool_name, tool_was_correct and was_correct)
        
        return self.trust_manager.get_all_weights()
    
    def get_tool_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics on all tools' historical performance.
        
        Useful for understanding which tools are trustworthy.
        
        Returns:
            Dict mapping tool names to their statistics
        """
        return self.trust_manager.get_all_stats()
    
    def reset_tool_trust(self, tool_name: Optional[str] = None) -> None:
        """
        Reset tool trust weights.
        
        Args:
            tool_name: Specific tool to reset, or None to reset all
        """
        if tool_name:
            self.trust_manager.reset_tool(tool_name)
        else:
            self.trust_manager.reset_all()


def generate_conflict_report(conflicts: List[Conflict], resolutions: List[Dict[str, Any]] = None) -> str:
    """
    Generate human-readable conflict report.
    
    Args:
        conflicts: List of detected conflicts
        resolutions: Optional list of resolutions (aligned with conflicts)
        
    Returns:
        Formatted report string
    """
    if not conflicts:
        return "✅ No conflicts detected - all tools agree"
    
    report = "⚠️  CONFLICT DETECTION REPORT\n"
    report += "=" * 60 + "\n"
    report += f"Detected {len(conflicts)} conflict(s)\n"
    report += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, conflict in enumerate(conflicts, 1):
        report += f"Conflict #{i} - {conflict.severity.upper()} SEVERITY\n"
        report += "-" * 60 + "\n"
        report += f"Type: {conflict.conflict_type}\n"
        report += f"Finding: {conflict.finding}\n"
        report += f"Tools: {', '.join(conflict.tools_involved)}\n"
        
        # Show disagreement details
        for tool, value, conf in zip(conflict.tools_involved, conflict.values, conflict.confidences):
            report += f"  • {tool}: {value} (confidence: {conf:.1%})\n"
        
        report += f"Recommendation: {conflict.recommendation}\n"
        
        # Add resolution if available
        if resolutions and i <= len(resolutions):
            res = resolutions[i-1]
            report += "\nResolution:\n"
            report += f"  Decision: {res.get('decision', 'N/A')}\n"
            report += f"  Selected: {res.get('selected_tool', 'N/A')}\n"
            report += f"  Confidence: {res.get('confidence', 0):.1%}\n"
            report += f"  Reasoning: {res.get('reasoning', 'N/A')}\n"
            if res.get('should_defer', False):
                report += "  ⚠️  FLAGGED FOR HUMAN REVIEW\n"
        
        report += "\n"
    
    return report
