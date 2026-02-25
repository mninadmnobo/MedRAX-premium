"""
Argumentation Graph: Structures and analyzes conflicts as explicit argument graphs.

Instead of just looking at confidence numbers, this module builds a graph showing:
- Which tools support a claim (support edges)
- Which tools attack a claim (attack edges)
- Whether there are circular arguments (cycles)
- How clear the winner is (certainty score)

This provides explainability: radiologists can see WHY tools disagree.
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class ArgumentNode:
    """One tool's position on a claim"""
    tool_name: str
    confidence: float  # Tool's confidence [0, 1]
    is_support: bool   # True = supports claim, False = attacks claim
    strength: float    # Weighted strength (confidence * trust_weight)
    
    def __repr__(self):
        position = "SUPPORT" if self.is_support else "ATTACK"
        return f"{self.tool_name}:{self.confidence:.2f}({position})"


@dataclass
class ArgumentGraph:
    """Full argumentation graph for a conflict"""
    claim: str                          # What we're arguing about (e.g., "Cardiomegaly present")
    support_nodes: List[ArgumentNode]   # Tools/evidence supporting the claim
    attack_nodes: List[ArgumentNode]    # Tools/evidence attacking the claim
    
    # Computed properties
    support_strength: float             # Sum of support confidences
    attack_strength: float              # Sum of attack confidences
    has_cycles: bool                    # Circular logic detected?
    certainty: float                    # 0.0 = completely unclear, 1.0 = very certain
    net_winner: str                     # "support", "attack", or "unclear"
    confidence_gap: float               # |support - attack|
    
    def __repr__(self):
        return (
            f"ArgumentGraph(claim='{self.claim}', "
            f"support={self.support_strength:.2f}, attack={self.attack_strength:.2f}, "
            f"winner={self.net_winner}, certainty={self.certainty:.2f})"
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/visualization"""
        return {
            "claim": self.claim,
            "support_nodes": [
                {
                    "tool": n.tool_name,
                    "confidence": n.confidence,
                    "strength": n.strength
                }
                for n in self.support_nodes
            ],
            "attack_nodes": [
                {
                    "tool": n.tool_name,
                    "confidence": n.confidence,
                    "strength": n.strength
                }
                for n in self.attack_nodes
            ],
            "support_strength": self.support_strength,
            "attack_strength": self.attack_strength,
            "has_cycles": self.has_cycles,
            "certainty": self.certainty,
            "net_winner": self.net_winner,
            "confidence_gap": self.confidence_gap,
        }


class ArgumentGraphBuilder:
    """
    Builds argument graphs from conflict data.
    
    Takes tool outputs and confidence scores, structures them as an argument graph.
    """
    
    def build_from_conflict(
        self,
        claim: str,
        tools_involved: List[str],
        confidences: List[float],
        values: List,
        tool_trust_weights: Optional[Dict[str, float]] = None
    ) -> ArgumentGraph:
        """
        Build an argument graph from conflict data.
        
        Args:
            claim: What we're arguing about (e.g., "Cardiomegaly present")
            tools_involved: List of tool names
            confidences: Confidence scores from each tool [0, 1]
            values: The actual outputs/values from each tool
            tool_trust_weights: Optional trust weights for each tool (default 1.0)
            
        Returns:
            ArgumentGraph showing support/attack structure
        """
        if not tool_trust_weights:
            tool_trust_weights = dict.fromkeys(tools_involved, 1.0)
        
        # Determine support vs attack based on confidence threshold and values
        support_nodes = []
        attack_nodes = []
        
        for tool, confidence, value in zip(tools_involved, confidences, values):
            trust_weight = tool_trust_weights.get(tool, 1.0)
            strength = confidence * trust_weight
            
            # Heuristic: >0.5 confidence = supports presence, <0.5 = attacks
            is_support = confidence > 0.5
            
            node = ArgumentNode(
                tool_name=tool,
                confidence=confidence,
                is_support=is_support,
                strength=strength
            )
            
            if is_support:
                support_nodes.append(node)
            else:
                attack_nodes.append(node)
        
        # Calculate strengths
        support_strength = sum(n.strength for n in support_nodes)
        attack_strength = sum(n.strength for n in attack_nodes)
        
        # Detect cycles (simplified: check for conflicting positions)
        has_cycles = self._detect_cycles(support_nodes, attack_nodes)
        
        # Calculate certainty (how clear is the winner?)
        confidence_gap = abs(support_strength - attack_strength)
        total_strength = support_strength + attack_strength
        
        if total_strength == 0:
            certainty = 0.0
        else:
            # Certainty = how dominant the winner is
            max_strength = max(support_strength, attack_strength)
            certainty = max_strength / total_strength if total_strength > 0 else 0.0
        
        # Determine winner
        if support_strength > attack_strength * 1.1:  # >10% gap
            net_winner = "support"
        elif attack_strength > support_strength * 1.1:
            net_winner = "attack"
        else:
            net_winner = "unclear"
        
        graph = ArgumentGraph(
            claim=claim,
            support_nodes=support_nodes,
            attack_nodes=attack_nodes,
            support_strength=support_strength,
            attack_strength=attack_strength,
            has_cycles=has_cycles,
            certainty=certainty,
            net_winner=net_winner,
            confidence_gap=confidence_gap,
        )
        
        return graph
    
    def _detect_cycles(
        self,
        support_nodes: List[ArgumentNode],
        attack_nodes: List[ArgumentNode]
    ) -> bool:
        """
        Detect circular logic patterns.
        
        Simplified implementation: if we have both strong support and strong attack,
        it might indicate contradictory interpretations (cycle-like).
        
        Args:
            support_nodes: Nodes supporting the claim
            attack_nodes: Nodes attacking the claim
            
        Returns:
            True if cycles detected, False otherwise
        """
        # Simple heuristic: if both sides have similar strength and both are present,
        # it's somewhat contradictory (circular-ish)
        if not support_nodes or not attack_nodes:
            return False
        
        support_strength = sum(n.strength for n in support_nodes)
        attack_strength = sum(n.strength for n in attack_nodes)
        
        # If both sides are equally strong, it's contradictory/circular
        if abs(support_strength - attack_strength) < 0.3:
            return True
        
        return False
    
    def analyze_conflict_clarity(self, graph: ArgumentGraph) -> Dict[str, any]:
        """
        Analyze how clear/unclear the conflict is.
        
        Returns metrics about the quality of the graph for decision-making.
        """
        return {
            "clarity_score": graph.certainty,
            "has_ambiguity": graph.has_cycles,
            "support_count": len(graph.support_nodes),
            "attack_count": len(graph.attack_nodes),
            "gap_magnitude": graph.confidence_gap,
            "net_winner": graph.net_winner,
        }


class ArgumentGraphVisualizer:
    """
    Helper class to visualize argument graphs for debugging/reporting.
    """
    
    @staticmethod
    def to_text(graph: ArgumentGraph) -> str:
        """
        Convert graph to human-readable text format.
        
        Example output:
        ```
        CLAIM: "Cardiomegaly present"
        
        SUPPORT (strength: 2.42):
          ├─ DenseNet: 0.92 (trusted)
          ├─ Segmentation: 0.85 (reliable)
          └─ CheXpert: 0.65 (moderate)
        
        ATTACK (strength: 0.30):
          └─ LLaVA: 0.30 (less trusted)
        
        ANALYSIS:
          Winner: SUPPORT (gap: 2.12)
          Certainty: 0.89
          Cycles: None
        ```
        """
        lines = []
        lines.append(f"CLAIM: \"{graph.claim}\"")
        lines.append("")
        
        # Support side
        lines.append(f"SUPPORT (strength: {graph.support_strength:.2f}):")
        if graph.support_nodes:
            for i, node in enumerate(graph.support_nodes):
                connector = "├─" if i < len(graph.support_nodes) - 1 else "└─"
                lines.append(f"  {connector} {node.tool_name}: {node.confidence:.2f} "
                           f"(strength: {node.strength:.2f})")
        else:
            lines.append("  (none)")
        lines.append("")
        
        # Attack side
        lines.append(f"ATTACK (strength: {graph.attack_strength:.2f}):")
        if graph.attack_nodes:
            for i, node in enumerate(graph.attack_nodes):
                connector = "├─" if i < len(graph.attack_nodes) - 1 else "└─"
                lines.append(f"  {connector} {node.tool_name}: {node.confidence:.2f} "
                           f"(strength: {node.strength:.2f})")
        else:
            lines.append("  (none)")
        lines.append("")
        
        # Analysis
        lines.append("ANALYSIS:")
        lines.append(f"  Winner: {graph.net_winner.upper()} (gap: {graph.confidence_gap:.2f})")
        lines.append(f"  Certainty: {graph.certainty:.2f}")
        lines.append(f"  Cycles: {'Yes' if graph.has_cycles else 'None'}")
        
        return "\n".join(lines)
