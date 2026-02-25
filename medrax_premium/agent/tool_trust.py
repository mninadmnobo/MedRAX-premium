"""
Tool Trust System: Track and learn tool reliability over time.

Each tool gets a trust weight based on historical performance:
- Track: how many times was it correct vs wrong?
- Learn: update after each resolved case
- Persist: save weights to JSON so they survive restarts
- Use: weight tool opinions by their reliability

Example:
  DenseNet: 92 correct / 100 total = 0.92 weight (very trusted)
  LLaVA: 71 correct / 100 total = 0.71 weight (moderately trusted)
"""

import json
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ToolTrust:
    """
    Trust record for a single tool.
    
    Tracks: correct_count / total_count = weight
    """
    tool_name: str
    correct_count: int = 0          # How many times was it correct?
    total_count: int = 0            # How many total predictions?
    weight: float = 1.0             # correct_count / total_count
    last_updated: str = ""          # When was it last updated?
    
    def update(self, was_correct: bool) -> None:
        """
        Update trust after a resolved case.
        
        Args:
            was_correct: True if this tool's prediction was correct
        """
        self.total_count += 1
        if was_correct:
            self.correct_count += 1
        
        # Recalculate weight
        if self.total_count > 0:
            self.weight = self.correct_count / self.total_count
        
        self.last_updated = datetime.now().isoformat()
    
    def get_weight(self) -> float:
        """Get current trust weight [0, 1]"""
        return self.weight
    
    def accuracy(self) -> float:
        """Get accuracy percentage"""
        if self.total_count == 0:
            return 0.0
        return (self.correct_count / self.total_count) * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class ToolTrustManager:
    """
    Manages trust weights for all tools.
    
    Responsibilities:
    - Initialize trust weights (uniform or from file)
    - Get current weight for a tool
    - Update trust after resolving cases
    - Persist weights to disk
    - Provide weighted voting mechanism
    """
    
    def __init__(self, persistence_file: Optional[str] = None):
        """
        Initialize tool trust manager.
        
        Args:
            persistence_file: JSON file to save/load weights
                             If None, weights only exist in memory
        """
        self.tools: Dict[str, ToolTrust] = {}
        self.persistence_file = persistence_file
        
        # Load from file if it exists
        if persistence_file and os.path.exists(persistence_file):
            self.load_from_file()
    
    def initialize_tool(self, tool_name: str, initial_weight: float = 1.0) -> None:
        """
        Initialize a tool with a starting weight.
        
        Args:
            tool_name: Name of the tool
            initial_weight: Starting trust weight [0, 1], default 1.0 (neutral)
        """
        if tool_name not in self.tools:
            # Back-calculate correct_count to match the weight
            # If weight = 0.8 and we want 10 initial samples:
            # correct_count = 0.8 * 10 = 8
            initial_samples = 10
            self.tools[tool_name] = ToolTrust(
                tool_name=tool_name,
                correct_count=int(initial_weight * initial_samples),
                total_count=initial_samples,
                weight=initial_weight,
                last_updated=datetime.now().isoformat()
            )
    
    def get_weight(self, tool_name: str) -> float:
        """
        Get current trust weight for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Trust weight [0, 1]. Default 1.0 if tool not yet tracked.
        """
        if tool_name not in self.tools:
            self.initialize_tool(tool_name)
        
        return self.tools[tool_name].get_weight()
    
    def get_all_weights(self) -> Dict[str, float]:
        """Get weights for all tools"""
        return {name: trust.get_weight() for name, trust in self.tools.items()}
    
    def update_trust(self, tool_name: str, was_correct: bool) -> None:
        """
        Update trust weight after a resolved case.
        
        Args:
            tool_name: Name of the tool
            was_correct: True if prediction was correct, False if wrong
        """
        if tool_name not in self.tools:
            self.initialize_tool(tool_name)
        
        self.tools[tool_name].update(was_correct)
        
        # Auto-save to file if configured
        if self.persistence_file:
            self.save_to_file()
    
    def weighted_vote(
        self,
        tools_and_confidences: list[Tuple[str, float]]
    ) -> float:
        """
        Combine tool outputs weighted by trust.
        
        Instead of simple average: (conf1 + conf2) / 2
        Uses weighted average: (conf1 * weight1 + conf2 * weight2) / (weight1 + weight2)
        
        Example:
          Tools and their confidences:
            - DenseNet: 0.92 confidence, 0.92 trust weight
            - LLaVA: 0.30 confidence, 0.71 trust weight
          
          Old average: (0.92 + 0.30) / 2 = 0.61 (treats equally)
          New weighted: (0.92 * 0.92 + 0.30 * 0.71) / (0.92 + 0.71)
                      = (0.85 + 0.21) / 1.63
                      = 1.06 / 1.63 = 0.65 (favors reliable tool)
        
        Args:
            tools_and_confidences: List of (tool_name, confidence) tuples
            
        Returns:
            Weighted confidence score [0, 1]
        """
        if not tools_and_confidences:
            return 0.0
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for tool_name, confidence in tools_and_confidences:
            weight = self.get_weight(tool_name)
            weighted_sum += confidence * weight
            weight_sum += weight
        
        if weight_sum == 0:
            return 0.0
        
        return weighted_sum / weight_sum
    
    def get_tool_stats(self, tool_name: str) -> Dict:
        """
        Get detailed statistics for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary with stats
        """
        if tool_name not in self.tools:
            self.initialize_tool(tool_name)
        
        trust = self.tools[tool_name]
        return {
            "tool_name": tool_name,
            "weight": trust.get_weight(),
            "correct_count": trust.correct_count,
            "total_count": trust.total_count,
            "accuracy_percent": trust.accuracy(),
            "last_updated": trust.last_updated,
        }
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all tools"""
        return {name: self.get_tool_stats(name) for name in self.tools.keys()}
    
    def save_to_file(self) -> None:
        """
        Save all trust weights to JSON file.
        
        This persists learning across runs.
        """
        if not self.persistence_file:
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.persistence_file) or ".", exist_ok=True)
        
        data = {
            "saved_at": datetime.now().isoformat(),
            "tools": {name: trust.to_dict() for name, trust in self.tools.items()}
        }
        
        with open(self.persistence_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self) -> None:
        """
        Load trust weights from JSON file.
        
        This restores learning from previous runs.
        """
        if not self.persistence_file or not os.path.exists(self.persistence_file):
            return
        
        try:
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)
            
            for tool_name, tool_data in data.get("tools", {}).items():
                self.tools[tool_name] = ToolTrust(
                    tool_name=tool_data["tool_name"],
                    correct_count=tool_data["correct_count"],
                    total_count=tool_data["total_count"],
                    weight=tool_data["weight"],
                    last_updated=tool_data["last_updated"]
                )
        except Exception as e:
            print(f"Warning: Could not load trust weights from {self.persistence_file}: {e}")
    
    def reset_tool(self, tool_name: str) -> None:
        """Reset a tool's trust to neutral (1.0)"""
        if tool_name in self.tools:
            self.tools[tool_name] = ToolTrust(
                tool_name=tool_name,
                correct_count=10,
                total_count=10,
                weight=1.0,
                last_updated=datetime.now().isoformat()
            )
            if self.persistence_file:
                self.save_to_file()
    
    def reset_all(self) -> None:
        """Reset all tools to neutral"""
        self.tools.clear()
        if self.persistence_file:
            self.save_to_file()
