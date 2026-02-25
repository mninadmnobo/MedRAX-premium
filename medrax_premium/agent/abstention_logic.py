"""
Abstention Logic: Decide when to abstain from making a conflict resolution.

Instead of always picking a winner, this module knows when to say "I don't know,
needs human review" based on multiple signals:

1. Circular logic detected (graph has cycles)
2. Vote too close (support vs attack strengths are similar)
3. Uncertainty too high (overall confidence in decision is low)
4. Clinical severity matters (be more careful with critical conditions)

This makes the system clinically safer by not forcing decisions when unclear.
"""

from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AbstentionReason(Enum):
    """Reasons why we might abstain from making a decision"""
    NONE = "proceed_with_confidence"
    CYCLE_DETECTED = "circular_logic_in_evidence"
    CLOSE_VOTE = "support_attack_strengths_too_similar"
    HIGH_UNCERTAINTY = "overall_confidence_too_low"
    CRITICAL_CONDITION_UNCLEAR = "life_threatening_finding_unclear"
    INSUFFICIENT_DATA = "not_enough_tools_reporting"


@dataclass
class AbstentionDecision:
    """Result of abstention analysis"""
    should_abstain: bool              # True = abstain, False = proceed with decision
    reason: AbstentionReason
    confidence: float                 # How confident are we in the decision (if not abstaining)?
    explanation: str                  # Human-readable explanation
    risk_level: str                   # "low", "medium", "high" risk of wrong decision
    
    def to_dict(self):
        return {
            "should_abstain": self.should_abstain,
            "reason": self.reason.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "risk_level": self.risk_level,
        }


class AbstentionLogic:
    """
    Determine when to abstain from conflict resolution.
    
    Thresholds are configurable for different clinical contexts.
    """
    
    # Thresholds for abstention decisions
    CLOSE_VOTE_THRESHOLD = 0.2         # If gap < 0.2, vote is "too close"
    UNCERTAINTY_THRESHOLD = 0.6        # If certainty < 0.6, "too uncertain"
    CRITICAL_CERTAINTY_THRESHOLD = 0.8 # Higher bar for critical conditions
    MIN_TOOLS_REPORTING = 2            # Need at least 2 tools to make decision
    
    def __init__(
        self,
        close_vote_thr: float = 0.2,
        uncertainty_thr: float = 0.6,
        critical_certainty_thr: float = 0.8,
        min_tools: int = 2
    ):
        """
        Initialize abstention logic with custom thresholds.
        
        Args:
            close_vote_thr: If gap between sides < this, vote is "too close"
            uncertainty_thr: If certainty < this, abstain
            critical_certainty_thr: Higher bar for life-threatening findings
            min_tools: Minimum tools needed to make decision
        """
        self._close_vote_threshold = close_vote_thr
        self._uncertainty_threshold = uncertainty_thr
        self._critical_certainty_threshold = critical_certainty_thr
        self._min_tools = min_tools
    
    def should_abstain(
        self,
        support_strength: float,
        attack_strength: float,
        certainty: float,
        has_cycles: bool,
        clinical_severity: str = "moderate",
        num_tools: int = 2,
        bert_contradiction_prob: float = 0.0
    ) -> AbstentionDecision:
        """
        Main decision: should we abstain from making a resolution?
        
        Checks multiple conditions to decide if the conflict is too unclear/risky.
        
        Args:
            support_strength: Combined strength of tools supporting the claim
            attack_strength: Combined strength of tools attacking the claim
            certainty: Overall certainty [0, 1] that we have the right answer
            has_cycles: True if circular logic detected in argument graph
            clinical_severity: "critical" (life-threatening), "moderate", "minor"
            num_tools: How many tools are reporting on this finding?
            bert_contradiction_prob: BERT's confidence that there's real contradiction
            
        Returns:
            AbstentionDecision with reasoning
        """
        
        # Check 1: Insufficient data
        if num_tools < self._min_tools:
            return AbstentionDecision(
                should_abstain=True,
                reason=AbstentionReason.INSUFFICIENT_DATA,
                confidence=0.0,
                explanation=f"Only {num_tools} tool(s) reporting. Need at least {self._min_tools}.",
                risk_level="high"
            )
        
        # Check 2: Circular logic detected
        if has_cycles:
            return AbstentionDecision(
                should_abstain=True,
                reason=AbstentionReason.CYCLE_DETECTED,
                confidence=0.0,
                explanation="Circular logic detected in argument graph. Tools are stuck in contradictory interpretations.",
                risk_level="high"
            )
        
        # Check 3: Vote too close (too similar strength on both sides)
        gap = abs(support_strength - attack_strength)
        if gap < self._close_vote_threshold:
            return AbstentionDecision(
                should_abstain=True,
                reason=AbstentionReason.CLOSE_VOTE,
                confidence=certainty,
                explanation=(
                    f"Evidence is split: support={support_strength:.2f}, "
                    f"attack={attack_strength:.2f}. Gap={gap:.2f} is too small to decide safely."
                ),
                risk_level="high"
            )
        
        # Check 4: Uncertainty too high
        threshold = self._uncertainty_threshold
        if clinical_severity == "critical":
            threshold = self._critical_certainty_threshold
        
        if certainty < threshold:
            reason = (
                AbstentionReason.CRITICAL_CONDITION_UNCLEAR
                if clinical_severity == "critical"
                else AbstentionReason.HIGH_UNCERTAINTY
            )
            
            critical_msg = "Critical condition " if clinical_severity == "critical" else ""
            
            return AbstentionDecision(
                should_abstain=True,
                reason=reason,
                confidence=certainty,
                explanation=(
                    f"Uncertainty too high (certainty={certainty:.2f}, "
                    f"threshold={threshold:.2f}). "
                    f"{critical_msg}Requires human review."
                ),
                risk_level="high" if clinical_severity == "critical" else "medium"
            )
        
        # All checks passed - proceed with decision
        return AbstentionDecision(
            should_abstain=False,
            reason=AbstentionReason.NONE,
            confidence=certainty,
            explanation=f"Sufficient clarity (certainty={certainty:.2f}) with clear evidence.",
            risk_level="low"
        )
    
    def assess_risk_level(
        self,
        certainty: float,
        gap: float,
        clinical_severity: str,
        bert_contradiction: float = 0.0
    ) -> str:
        """
        Assess overall risk level of the decision.
        
        Combines multiple factors to give radiologist a risk assessment.
        
        Args:
            certainty: 0.0 = very uncertain, 1.0 = very certain
            gap: Difference between support and attack strength
            clinical_severity: "critical", "moderate", "minor"
            bert_contradiction: BERT's contradiction confidence
            
        Returns:
            "low", "medium", or "high" risk
        """
        risk_score = 0.0
        
        # Certainty factor (40% of risk)
        if certainty < 0.6:
            risk_score += 0.4
        elif certainty < 0.75:
            risk_score += 0.2
        
        # Gap factor (30% of risk)
        if gap < 0.2:
            risk_score += 0.3
        elif gap < 0.4:
            risk_score += 0.15
        
        # Clinical severity (20% of risk)
        if clinical_severity == "critical":
            risk_score += 0.2
        elif clinical_severity == "moderate":
            risk_score += 0.1
        
        # BERT contradiction (10% of risk)
        if bert_contradiction > 0.8:
            risk_score += 0.1
        elif bert_contradiction > 0.6:
            risk_score += 0.05
        
        # Convert score to risk level
        if risk_score >= 0.5:
            return "high"
        elif risk_score >= 0.25:
            return "medium"
        else:
            return "low"
    
    def explain_decision(self, decision: AbstentionDecision) -> str:
        """
        Generate human-readable explanation of abstention decision.
        
        Useful for logging and radiologist review.
        """
        if decision.should_abstain:
            return (
                f"⚠️  ABSTAINING FROM DECISION\n"
                f"Reason: {decision.reason.value}\n"
                f"Risk Level: {decision.risk_level.upper()}\n"
                f"Details: {decision.explanation}\n"
                f"Recommendation: Review by radiologist required."
            )
        else:
            return (
                f"✅ PROCEEDING WITH DECISION\n"
                f"Confidence: {decision.confidence:.1%}\n"
                f"Risk Level: {decision.risk_level.upper()}\n"
                f"Details: {decision.explanation}"
            )
