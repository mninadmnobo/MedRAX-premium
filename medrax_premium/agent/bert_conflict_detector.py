"""
BERT-based Conflict Detection for MedRAX tool outputs.

Uses a fine-tuned BERT model to detect semantic conflicts between
textual outputs from different medical imaging analysis tools.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)
import numpy as np


@dataclass
class ConflictPrediction:
    """Result of BERT conflict detection with full probability scores."""
    has_conflict: bool
    conflict_probability: float  # Contradiction probability (main score)
    conflict_type: str  # "contradiction", "agreement", "neutral"
    explanation: str
    text_pair: Tuple[str, str]
    # Additional scores for resolution pipeline
    entailment_prob: float = 0.0  # Agreement probability
    neutral_prob: float = 0.0  # Neutral probability
    raw_logits: Optional[List[float]] = None  # Raw model logits
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_conflict": self.has_conflict,
            "conflict_probability": self.conflict_probability,
            "entailment_prob": self.entailment_prob,
            "neutral_prob": self.neutral_prob,
            "conflict_type": self.conflict_type,
            "explanation": self.explanation,
            "text_pair": self.text_pair,
        }
    
    def get_all_scores(self) -> Dict[str, float]:
        """Get all probability scores for resolution pipeline."""
        return {
            "contradiction_prob": self.conflict_probability,
            "entailment_prob": self.entailment_prob,
            "neutral_prob": self.neutral_prob,
        }
    

class BERTConflictDetector:
    """
    BERT-based conflict detection for medical text outputs.
    
    Uses Natural Language Inference (NLI) approach:
    - Given two statements from different tools, classify as:
      - ENTAILMENT (agreement): Tools say the same thing
      - CONTRADICTION (conflict): Tools disagree
      - NEUTRAL: Statements are unrelated or complementary
    
    Can use pre-trained medical NLI models or fine-tuned BERT.
    """
    
    # Conflict type mappings for NLI models
    NLI_LABELS = {
        0: "contradiction",  # CONFLICT
        1: "neutral",        # No conflict, different aspects
        2: "entailment",     # Agreement
    }
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        nli_model_name: str = "microsoft/deberta-base-mnli",
        device: Optional[str] = None,
        conflict_threshold: float = 0.7,
        use_nli: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the BERT conflict detector.
        
        Args:
            model_name: Base BERT model for embeddings (used if not using NLI)
            nli_model_name: Pre-trained NLI model for conflict detection
            device: Device to run on (cuda/cpu)
            conflict_threshold: Probability threshold to declare conflict
            use_nli: Whether to use NLI approach (recommended) or embedding similarity
            cache_dir: Directory to cache downloaded models
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conflict_threshold = conflict_threshold
        self.use_nli = use_nli
        self.cache_dir = cache_dir
        
        if use_nli:
            # Load NLI model for conflict detection
            print(f"Loading NLI model: {nli_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                nli_model_name, cache_dir=cache_dir
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                nli_model_name, cache_dir=cache_dir
            ).to(self.device)
            self.model.eval()
        else:
            # Load BERT for embedding-based similarity
            print(f"Loading BERT model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                model_name, cache_dir=cache_dir
            ).to(self.device)
            self.model.eval()
            
            # Conflict classifier head for embedding approach
            self.conflict_classifier = nn.Sequential(
                nn.Linear(768 * 3, 256),  # [emb1, emb2, |emb1-emb2|]
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 3),  # conflict, neutral, agreement
            ).to(self.device)
    
    def _format_tool_output(self, tool_name: str, output: Any) -> str:
        """
        Convert tool output to a natural language statement.
        
        Args:
            tool_name: Name of the tool
            output: Raw output from the tool
            
        Returns:
            Formatted text statement
        """
        if isinstance(output, dict):
            # Handle different tool output formats
            if "findings" in output:
                return f"The {tool_name} found: {output['findings']}"
            elif "prediction" in output:
                return f"The {tool_name} predicts: {output['prediction']}"
            elif "report" in output:
                return output["report"]
            elif "text" in output:
                return output["text"]
            elif "description" in output:
                return output["description"]
            else:
                # Generic dict formatting
                return f"The {tool_name} reports: {str(output)}"
        elif isinstance(output, str):
            return output
        elif isinstance(output, (int, float)):
            return f"The {tool_name} confidence is {output:.2%}"
        else:
            return str(output)
    
    def _detect_conflict_nli(
        self, 
        text1: str, 
        text2: str
    ) -> ConflictPrediction:
        """
        Detect conflict using NLI (Natural Language Inference).
        
        Treats text1 as premise and text2 as hypothesis.
        
        Args:
            text1: First tool's output text
            text2: Second tool's output text
            
        Returns:
            ConflictPrediction with results
        """
        # Tokenize the pair
        inputs = self.tokenizer(
            text1, 
            text2, 
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
        
        # Get predicted class and probabilities
        pred_class = torch.argmax(probs).item()
        conflict_type = self.NLI_LABELS.get(pred_class, "neutral")
        
        # Contradiction probability
        contradiction_prob = probs[0].item()
        entailment_prob = probs[2].item()
        neutral_prob = probs[1].item()
        
        # Determine if there's a conflict
        has_conflict = (
            conflict_type == "contradiction" and 
            contradiction_prob >= self.conflict_threshold
        )
        
        # Generate explanation
        if has_conflict:
            explanation = (
                f"BERT NLI detected contradiction with {contradiction_prob:.1%} confidence. "
                f"The tool outputs appear to disagree on the findings."
            )
        elif conflict_type == "entailment":
            explanation = (
                f"BERT NLI detected agreement with {entailment_prob:.1%} confidence. "
                f"The tool outputs are consistent."
            )
        else:
            explanation = (
                f"BERT NLI found neutral relationship ({neutral_prob:.1%}). "
                f"The outputs discuss different aspects or are unrelated."
            )
        
        return ConflictPrediction(
            has_conflict=has_conflict,
            conflict_probability=contradiction_prob,
            conflict_type=conflict_type,
            explanation=explanation,
            text_pair=(text1, text2),
            entailment_prob=entailment_prob,
            neutral_prob=neutral_prob,
            raw_logits=logits[0].cpu().tolist(),
        )
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get BERT embedding for text (CLS token)."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding
    
    def _detect_conflict_embedding(
        self, 
        text1: str, 
        text2: str
    ) -> ConflictPrediction:
        """
        Detect conflict using embedding similarity.
        
        Uses cosine similarity and learned classifier.
        
        Args:
            text1: First tool's output text
            text2: Second tool's output text
            
        Returns:
            ConflictPrediction with results
        """
        # Get embeddings
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        # Compute similarity
        cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        
        # Concatenate features: [emb1, emb2, |emb1-emb2|]
        diff = torch.abs(emb1 - emb2)
        combined = torch.cat([emb1, emb2, diff], dim=-1)
        
        # Classify
        with torch.no_grad():
            logits = self.conflict_classifier(combined)
            probs = torch.softmax(logits, dim=-1)[0]
        
        pred_class = torch.argmax(probs).item()
        conflict_types = ["contradiction", "neutral", "entailment"]
        conflict_type = conflict_types[pred_class]
        
        contradiction_prob = probs[0].item()
        has_conflict = (
            conflict_type == "contradiction" and 
            contradiction_prob >= self.conflict_threshold
        )
        
        explanation = (
            f"Embedding similarity: {cosine_sim:.2f}. "
            f"Conflict probability: {contradiction_prob:.1%}."
        )
        
        return ConflictPrediction(
            has_conflict=has_conflict,
            conflict_probability=contradiction_prob,
            conflict_type=conflict_type,
            explanation=explanation,
            text_pair=(text1, text2),
        )
    
    def detect_conflict(
        self,
        text1: str,
        text2: str,
        tool1_name: str = "Tool1",
        tool2_name: str = "Tool2",
    ) -> ConflictPrediction:
        """
        Detect if two tool outputs are in conflict.
        
        Args:
            text1: First tool's output (text or dict)
            text2: Second tool's output (text or dict)
            tool1_name: Name of first tool
            tool2_name: Name of second tool
            
        Returns:
            ConflictPrediction with conflict analysis
        """
        # Format outputs to text if needed
        if not isinstance(text1, str):
            text1 = self._format_tool_output(tool1_name, text1)
        if not isinstance(text2, str):
            text2 = self._format_tool_output(tool2_name, text2)
        
        # Use appropriate detection method
        if self.use_nli:
            return self._detect_conflict_nli(text1, text2)
        else:
            return self._detect_conflict_embedding(text1, text2)
    
    def detect_conflicts_batch(
        self,
        tool_outputs: List[Dict[str, Any]],
    ) -> List[ConflictPrediction]:
        """
        Detect conflicts among multiple tool outputs.
        
        Compares all pairs of outputs.
        
        Args:
            tool_outputs: List of dicts with 'tool_name' and 'output' keys
            
        Returns:
            List of ConflictPrediction for each pair with detected conflicts
        """
        conflicts = []
        n = len(tool_outputs)
        
        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                tool1 = tool_outputs[i]
                tool2 = tool_outputs[j]
                
                prediction = self.detect_conflict(
                    text1=tool1.get("output", ""),
                    text2=tool2.get("output", ""),
                    tool1_name=tool1.get("tool_name", f"Tool{i}"),
                    tool2_name=tool2.get("tool_name", f"Tool{j}"),
                )
                
                if prediction.has_conflict:
                    conflicts.append(prediction)
        
        return conflicts
    
    def analyze_finding_consistency(
        self,
        findings: List[Dict[str, Any]],
        finding_key: str = "description",
    ) -> Dict[str, Any]:
        """
        Analyze consistency of findings from multiple tools.
        
        Args:
            findings: List of finding dicts from different tools
            finding_key: Key to extract text from findings
            
        Returns:
            Analysis dict with conflicts and agreement scores
        """
        if len(findings) < 2:
            return {
                "has_conflicts": False,
                "conflicts": [],
                "agreement_score": 1.0,
                "message": "Need at least 2 findings to check consistency"
            }
        
        # Prepare tool outputs
        tool_outputs = []
        for f in findings:
            tool_outputs.append({
                "tool_name": f.get("source_tool", f.get("tool", "unknown")),
                "output": f.get(finding_key, f.get("text", str(f))),
            })
        
        # Detect conflicts
        conflicts = self.detect_conflicts_batch(tool_outputs)
        
        # Calculate agreement score
        total_pairs = len(findings) * (len(findings) - 1) // 2
        conflict_count = len(conflicts)
        agreement_score = 1.0 - (conflict_count / total_pairs) if total_pairs > 0 else 1.0
        
        return {
            "has_conflicts": len(conflicts) > 0,
            "conflicts": [
                {
                    "probability": c.conflict_probability,
                    "type": c.conflict_type,
                    "explanation": c.explanation,
                    "texts": c.text_pair,
                }
                for c in conflicts
            ],
            "agreement_score": agreement_score,
            "num_conflicts": conflict_count,
            "total_comparisons": total_pairs,
        }


class MedicalConflictDetector(BERTConflictDetector):
    """
    Specialized conflict detector for medical/radiology text.
    
    Uses domain-specific prompts and medical NLI understanding.
    """
    
    # Medical-specific contradiction patterns
    CONTRADICTION_PATTERNS = [
        ("present", "absent"),
        ("normal", "abnormal"),
        ("positive", "negative"),
        ("enlarged", "normal size"),
        ("opacification", "clear"),
        ("consolidation", "no consolidation"),
        ("effusion", "no effusion"),
        ("pneumothorax", "no pneumothorax"),
        ("cardiomegaly", "normal heart size"),
        ("edema", "no edema"),
    ]
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        nli_model_name: str = "microsoft/deberta-base-mnli",
        device: Optional[str] = None,
        conflict_threshold: float = 0.6,  # Lower threshold for medical text
        cache_dir: Optional[str] = None,
    ):
        """Initialize with medical-optimized settings."""
        super().__init__(
            model_name=model_name,
            nli_model_name=nli_model_name,
            device=device,
            conflict_threshold=conflict_threshold,
            use_nli=True,
            cache_dir=cache_dir,
        )
    
    def _check_pattern_contradiction(self, text1: str, text2: str) -> Optional[str]:
        """
        Quick check for obvious medical contradictions using patterns.
        
        Returns explanation if pattern-based contradiction found.
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for pos, neg in self.CONTRADICTION_PATTERNS:
            # Check if one text has positive and other has negative
            if pos in text1_lower and neg in text2_lower:
                return f"Pattern contradiction detected: '{pos}' vs '{neg}'"
            if neg in text1_lower and pos in text2_lower:
                return f"Pattern contradiction detected: '{neg}' vs '{pos}'"
        
        return None
    
    def detect_conflict(
        self,
        text1: str,
        text2: str,
        tool1_name: str = "Tool1",
        tool2_name: str = "Tool2",
    ) -> ConflictPrediction:
        """
        Detect medical conflicts with pattern pre-check.
        """
        # Format outputs
        if not isinstance(text1, str):
            text1 = self._format_tool_output(tool1_name, text1)
        if not isinstance(text2, str):
            text2 = self._format_tool_output(tool2_name, text2)
        
        # Quick pattern check first
        pattern_conflict = self._check_pattern_contradiction(text1, text2)
        
        # Run BERT NLI
        bert_result = self._detect_conflict_nli(text1, text2)
        
        # Combine results - boost confidence if pattern also detected
        if pattern_conflict:
            combined_prob = min(1.0, bert_result.conflict_probability + 0.2)
            has_conflict = combined_prob >= self.conflict_threshold
            explanation = f"{pattern_conflict}. {bert_result.explanation}"
            
            return ConflictPrediction(
                has_conflict=has_conflict,
                conflict_probability=combined_prob,
                conflict_type="contradiction" if has_conflict else bert_result.conflict_type,
                explanation=explanation,
                text_pair=(text1, text2),
            )
        
        return bert_result


# Convenience function for quick conflict detection
def detect_text_conflict(
    text1: str,
    text2: str,
    device: Optional[str] = None,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Quick function to detect conflict between two texts.
    
    Args:
        text1: First text
        text2: Second text
        device: Device to use (cuda/cpu)
        threshold: Conflict probability threshold
        
    Returns:
        Dict with conflict analysis
    """
    detector = BERTConflictDetector(
        device=device,
        conflict_threshold=threshold,
    )
    
    result = detector.detect_conflict(text1, text2)
    
    return {
        "has_conflict": result.has_conflict,
        "probability": result.conflict_probability,
        "type": result.conflict_type,
        "explanation": result.explanation,
    }
