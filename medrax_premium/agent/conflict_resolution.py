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
        
        # BERT detector (lazy loaded)
        self._bert_detector = None
        
        # Initialize GACL for semantic conflict detection
        self.gacl_detector = GACLConflictDetector(anomaly_threshold=0.6)
    
    @property
    def bert_detector(self):
        """Lazy load BERT detector on first use."""
        if self._bert_detector is None and self.use_bert:
            try:
                from .bert_conflict_detector import MedicalConflictDetector
                self._bert_detector = MedicalConflictDetector(
                    device=self.bert_device,
                    cache_dir=self.bert_cache_dir,
                    conflict_threshold=0.6,
                )
                print("✓ BERT conflict detector loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load BERT detector: {e}. Using rule-based detection.")
                self.use_bert = False
        return self._bert_detector
    
    def detect_conflicts(self, findings: List[CanonicalFinding]) -> List[Conflict]:
        """
        Analyze findings for conflicts using BERT + rule-based framework.
        
        Detection Pipeline:
        1. Group findings by pathology
        2. For each group with 2+ tools:
           a. BERT-based semantic conflict detection (if enabled)
           b. Rule-based presence/absence conflict detection
           c. GACL anatomical consistency check
        
        Args:
            findings: List of canonical findings from all tools
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Group findings by pathology
        by_pathology = self._group_by_pathology(findings)
        
        # Check each pathology for conflicts
        for pathology, tool_findings in by_pathology.items():
            # Need at least 2 tools to have a conflict
            if len(tool_findings) < 2:
                continue
            
            # STEP 1: BERT-based semantic conflict detection (PRIMARY)
            if self.use_bert:
                bert_conflicts = self._check_bert_conflicts(pathology, tool_findings)
                conflicts.extend(bert_conflicts)
            
            # STEP 2: Rule-based presence/absence conflicts (FALLBACK)
            presence_conflicts = self._check_presence_conflicts(pathology, tool_findings)
            # Only add if not already detected by BERT
            for pc in presence_conflicts:
                if not any(c.finding == pc.finding and c.conflict_type == "presence" for c in conflicts):
                    conflicts.append(pc)
            
            # STEP 3: GACL anatomical consistency check
            semantic_conflicts = self._check_semantic_conflicts_gacl(pathology, tool_findings)
            # Deduplicate: only add if not already detected by BERT for same tools
            for gc in semantic_conflicts:
                is_duplicate = any(
                    c.conflict_type == "semantic" and 
                    c.finding == gc.finding and
                    set(c.tools_involved) == set(gc.tools_involved)
                    for c in conflicts
                )
                if not is_duplicate:
                    conflicts.append(gc)
        
        return conflicts
    
    def _group_by_pathology(self, findings: List[CanonicalFinding]) -> Dict[str, List[CanonicalFinding]]:
        """Group findings by pathology name."""
        grouped = {}
        for finding in findings:
            if finding.pathology not in grouped:
                grouped[finding.pathology] = []
            grouped[finding.pathology].append(finding)
        return grouped
    
    def _check_presence_conflicts(self, pathology: str, findings: List[CanonicalFinding]) -> List[Conflict]:
        """
        Check if tools disagree on whether a finding is present or absent.
        
        Rules:
        - >0.7 confidence = present
        - <0.3 confidence = absent
        - 0.3-0.7 = uncertain
        
        Conflict if: max_confidence - min_confidence > threshold
        """
        conflicts = []
        
        confidences = [f.confidence for f in findings]
        max_conf = max(confidences)
        min_conf = min(confidences)
        
        # Check for significant disagreement
        if max_conf - min_conf > self.CONFIDENCE_GAP_THRESHOLD:
            # One tool says present, another says absent
            if max_conf > self.PRESENCE_THRESHOLD_HIGH and min_conf < self.PRESENCE_THRESHOLD_LOW:
                severity = "critical" if max_conf > 0.85 else "moderate"
                
                conflict = Conflict(
                    conflict_type="presence",
                    finding=pathology,
                    tools_involved=[f.source_tool for f in findings],
                    values=["present" if f.confidence > 0.5 else "absent" for f in findings],
                    confidences=confidences,
                    severity=severity,
                    recommendation=self._get_presence_resolution_strategy(findings)
                )
                conflicts.append(conflict)
        
        return conflicts
    
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
    
    def _check_semantic_conflicts_gacl(self, pathology: str, findings: List[CanonicalFinding]) -> List[Conflict]:
        """
        Detect semantic conflicts using GACL for ALL CXR pathologies.
        
        GACL (Graph-Based Anatomical Consistency Learning) detects when:
        - Segmentation finds anatomical patterns inconsistent with classification
        - Multiple tools contradict each other on disease presence
        - Anatomical measurements suggest pathology not detected by classifier
        
        Works for ALL pathologies: Pneumonia, Pneumothorax, Cardiomegaly, Effusion, etc.
        
        Args:
            pathology: The finding being checked (e.g., "Pneumonia", "Cardiomegaly")
            findings: Findings from different tools for this pathology
            
        Returns:
            List of conflicts (0 or 1) if semantic conflict detected
        """
        # Find segmentation and classification pairs
        seg_findings = [f for f in findings if "segmentation" in f.source_tool.lower()]
        clf_findings = [f for f in findings if "classifier" in f.source_tool.lower()]
        
        # Need both types to detect semantic conflict
        if not (seg_findings and clf_findings):
            return []
        
        # Use GACL for all pathologies
        try:
            for seg in seg_findings:
                for clf in clf_findings:
                    conflict = self.detect_semantic_conflict_gacl(
                        seg.raw_value,  # Segmentation measurements
                        clf.raw_value   # Classification prediction
                    )
                    if conflict:
                        return [conflict]
        except Exception as e:
            # Gracefully handle GACL failures
            print(f"  ⚠️ GACL analysis failed for {pathology}: {str(e)}")
        
        return []
    
    def detect_semantic_conflict_gacl(self, seg_output: Dict[str, Any], 
                                      classifier_output: Dict[str, Any]) -> Optional[Conflict]:
        """
        Detect semantic conflicts using Graph-Based Anatomical Consistency Learning.
        
        Detects when segmentation finds anatomical/structural anomalies but classifier 
        predicts normal (or vice versa). Uses learned anatomical patterns rather than 
        hardcoded rules.
        
        Works for ALL CXR pathologies:
        - Cardiac: Cardiomegaly, Enlarged Cardiomediastinum
        - Lung: Pneumonia, Pneumothorax, Consolidation
        - Pleural: Effusion, Pleural Thickening
        - Others: Nodule, Mass, Fracture, etc.
        
        Args:
            seg_output: Segmentation tool output with anatomical measurements
            classifier_output: Classification tool output
        
        Returns:
            Conflict object if semantic conflict detected, None otherwise
        """
        try:
            # Run GACL analysis
            analysis = self.gacl_detector.detect_semantic_conflict(seg_output, classifier_output)
            
            if not analysis.get("has_conflict", False):
                return None
            
            # Create conflict representation (generic for all pathologies)
            conflict = Conflict(
                conflict_type="semantic",
                finding="Anatomical pattern consistency",
                tools_involved=["segmentation_tool", "classification_model"],
                values=[
                    analysis["segmentation_pattern"],
                    analysis["classification_prediction"]
                ],
                confidences=[
                    1.0 - analysis["anomaly_score"],  # Abnormality confidence
                    analysis["classification_confidence"]
                ],
                severity="critical",
                recommendation=(
                    f"Graph-based anatomical analysis detected inconsistency. "
                    f"Segmentation analysis suggests: {analysis['most_likely_diagnosis']} "
                    f"({analysis['confidence_in_diagnosis']:.0%} confidence). "
                    f"Reason: {analysis['explanation']}"
                )
            )
            
            return conflict
        
        except Exception:
            # Gracefully handle GACL analysis failures
            return None
    
    def _check_bert_conflicts(
        self, 
        pathology: str, 
        findings: List[CanonicalFinding]
    ) -> List[Conflict]:
        """
        Use BERT to detect semantic conflicts in textual outputs.
        
        Compares all pairs of findings using NLI-based conflict detection.
        
        Args:
            pathology: The finding being checked
            findings: Findings from different tools
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        if not self.bert_detector:
            return conflicts
        
        # Compare all pairs of findings from DIFFERENT tools only.
        # Same tool called on different images is NOT a conflict.
        for i, finding1 in enumerate(findings):
            for finding2 in findings[i + 1:]:
                # Skip same-tool comparisons (e.g. classifier on image1 vs image2)
                if finding1.source_tool == finding2.source_tool:
                    continue
                # Extract text from findings
                text1 = self._extract_text_from_finding(finding1)
                text2 = self._extract_text_from_finding(finding2)
                
                if not text1 or not text2:
                    continue
                
                # Run BERT conflict detection
                try:
                    prediction = self.bert_detector.detect_conflict(
                        text1=text1,
                        text2=text2,
                        tool1_name=finding1.source_tool,
                        tool2_name=finding2.source_tool,
                    )
                    
                    if prediction.has_conflict:
                        # Determine severity based on confidence
                        if prediction.conflict_probability > 0.85:
                            severity = "critical"
                        elif prediction.conflict_probability > 0.7:
                            severity = "moderate"
                        else:
                            severity = "minor"
                        
                        # Capture ALL BERT scores for resolution pipeline
                        bert_scores = {
                            "contradiction_prob": prediction.conflict_probability,
                            "entailment_prob": prediction.entailment_prob,
                            "neutral_prob": prediction.neutral_prob,
                            "conflict_type": prediction.conflict_type,
                            "threshold_used": self.bert_detector.conflict_threshold,
                            "text1_preview": text1[:200],
                            "text2_preview": text2[:200],
                        }
                        
                        conflict = Conflict(
                            conflict_type="semantic",
                            finding=pathology,
                            tools_involved=[finding1.source_tool, finding2.source_tool],
                            values=[text1[:100], text2[:100]],  # Truncate for display
                            confidences=[finding1.confidence, finding2.confidence],
                            severity=severity,
                            recommendation=(
                                f"BERT detected contradiction ({prediction.conflict_probability:.0%} confidence). "
                                f"{prediction.explanation}"
                            ),
                            bert_scores=bert_scores,  # Include ALL BERT scores
                        )
                        conflicts.append(conflict)
                        
                except Exception as e:
                    print(f"  ⚠️ BERT conflict detection failed for {pathology}: {e}")
        
        return conflicts
    
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
            "primary": "segmentation_tool",
            "fallback": "phrase_grounding_tool",
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
        
        # ===== PREMIUM COMPONENT #1: BUILD ARGUMENT GRAPH =====
        argument_graph = None
        if self.enable_argumentation and len(findings) >= 1:
            try:
                # Get tool trust weights if enabled
                trust_weights = None
                if self.enable_tool_trust:
                    trust_weights = {
                        finding.source_tool: self.trust_manager.get_weight(finding.source_tool)
                        for finding in findings
                    }
                
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
            
            resolution = {
                "decision": "trust_primary_tool",
                "selected_tool": primary_tool,
                "value": primary_finding.confidence > 0.5,
                "confidence": adjusted_confidence,
                "reasoning": (
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
            
            resolution = {
                "decision": "trust_fallback_tool",
                "selected_tool": fallback_tool,
                "value": fallback_finding.confidence > 0.5,
                "confidence": adjusted_confidence,
                "reasoning": (
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
        
        # Calculate severity adjustment based on contradiction confidence
        if contradiction > self.BERT_HIGH_CONTRADICTION:
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
            return {
                "decision": "bert_requires_human_review",
                "selected_tool": None,
                "value": None,
                "confidence": 0.0,
                "reasoning": (
                    f"BERT detected strong contradiction ({bert_analysis['contradiction_prob']:.0%}), "
                    f"but tool confidences are too close ({highest.confidence:.0%} vs {second.confidence:.0%}). "
                    f"Deferring to radiologist review."
                ),
                "should_defer": True,
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
        Resolve conflict using confidence-weighted average.
        
        Args:
            findings: All findings for this pathology
            bert_analysis: Optional BERT analysis for additional context
            
        Returns:
            Resolution dict
        """
        # Weight by confidence
        total_weight = sum(f.confidence for f in findings)
        
        if total_weight == 0:
            return {
                "decision": "insufficient_confidence",
                "confidence": 0.0,
                "reasoning": "All tools have zero confidence",
                "should_defer": True,
                "bert_analysis": bert_analysis or {},
            }
        
        # Weighted average of "presence" (1 if confidence > 0.5, else 0)
        weighted_presence = sum(
            f.confidence * (1 if f.confidence > 0.5 else 0)
            for f in findings
        ) / total_weight
        
        avg_confidence = sum(f.confidence for f in findings) / len(findings)
        
        # Apply BERT adjustment if available
        if bert_analysis:
            avg_confidence *= bert_analysis.get("severity_adjustment", 1.0)
        
        return {
            "decision": "weighted_average",
            "value": weighted_presence > 0.5,
            "confidence": avg_confidence,
            "reasoning": f"Averaged {len(findings)} tool outputs with confidence weighting",
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
