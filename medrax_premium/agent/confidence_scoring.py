"""
Unified Confidence Scoring Pipeline for MedRAX
==============================================

A model-agnostic confidence scoring system that converts outputs from heterogeneous 
CXR AI tools into comparable calibrated confidence scores ∈ [0,1].

Key Features:
- Task-specific raw confidence extraction
- Min-max normalization across tools
- Isotonic regression / temperature scaling calibration
- Cross-model confidence fusion
- Validation metrics (ECE, Brier Score, AUROC)
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import math
import json
import pickle
from pathlib import Path
from collections import Counter
import warnings

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Calibration features will be limited.")


class TaskType(Enum):
    """Supported task types for confidence scoring."""
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    VQA = "vqa"
    GROUNDING = "grounding"
    REPORT = "report"
    GENERATION = "generation"
    UNKNOWN = "unknown"


@dataclass
class ModelOutput:
    """
    Standard output schema for all model outputs.
    
    Attributes:
        task_type: Type of task ("classification", "segmentation", etc.)
        raw_output: Original model output (logits, probs, text, masks, etc.)
        auxiliary: Additional data (logits, attention maps, masks, token_probs, etc.)
        model_name: Name of the model that produced this output
        timestamp: When this output was generated
    """
    task_type: str
    raw_output: Any
    auxiliary: Dict[str, Any] = field(default_factory=dict)
    model_name: str = "unknown"
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_type": self.task_type,
            "raw_output": str(self.raw_output)[:500],  # Truncate for safety
            "auxiliary_keys": list(self.auxiliary.keys()),
            "model_name": self.model_name,
            "timestamp": self.timestamp
        }


@dataclass
class ConfidenceResult:
    """
    Final confidence result schema with all scoring components.
    
    Attributes:
        task_type: Type of task
        raw_confidence: Original confidence before calibration
        uncertainty: Uncertainty measure ( 1 - confidence or entropy)
        calibrated_confidence: Final calibrated confidence ∈ [0,1]
        method: Method used for confidence extraction and calibration
        model_name: Name of the model
        metadata: Additional task-specific information
        is_calibrated: Whether calibration was applied
    """
    task_type: str
    raw_confidence: float
    uncertainty: float
    calibrated_confidence: float
    method: str
    model_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_calibrated: bool = False
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        # Ensure confidence bounds
        self.calibrated_confidence = max(0.0, min(1.0, self.calibrated_confidence))
        self.raw_confidence = max(0.0, min(1.0, self.raw_confidence))
        self.uncertainty = max(0.0, min(1.0, self.uncertainty))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BaseConfidenceExtractor(ABC):
    """
    Abstract base class for task-specific confidence extractors.
    
    Each task type implements its own extraction logic.
    """
    
    @abstractmethod
    def extract(self, model_output: ModelOutput) -> Tuple[float, float, Dict[str, Any]]:
        """
        Extract raw confidence and uncertainty from model output.
        
        Args:
            model_output: Standardized model output
            
        Returns:
            Tuple of (raw_confidence, uncertainty, extraction_metadata)
        """
        pass
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of the extraction method."""
        pass


class ClassificationConfidenceExtractor(BaseConfidenceExtractor):
    """
    Confidence extractor for classification models (DenseNet-121, etc.).
    
    Expects tool to return:
        metadata["confidence_data"]["probabilities"] = {pathology: probability}
    
    The probabilities are already sigmoid-applied by the tool.
    """
    
    def __init__(self, target_pathology: Optional[str] = None):
        """
        Initialize the extractor.
        
        Args:
            target_pathology: Specific pathology to extract confidence for (optional)
        """
        self.target_pathology = target_pathology
    
    @property
    def method_name(self) -> str:
        return "sigmoid_probability"
    
    def extract(self, model_output: ModelOutput) -> Tuple[float, float, Dict[str, Any]]:
        """
        Extract confidence from classification output.
        
        Primary path: Uses pre-computed probabilities from tool's confidence_data.
        Fallback: Extracts directly from raw_output dict if confidence_data unavailable.
        """
        auxiliary = model_output.auxiliary
        raw = model_output.raw_output
        
        # Primary path: Use pre-computed confidence_data from tool
        confidence_data = auxiliary.get("confidence_data")
        if confidence_data and isinstance(confidence_data, dict):
            probabilities = confidence_data.get("probabilities")
            if probabilities and isinstance(probabilities, dict):
                return self._extract_from_probabilities(probabilities, source="tool_confidence_data")
        
        # Fallback: Extract directly from raw_output (e.g., {pathology: prob} dict)
        if isinstance(raw, dict) and raw:
            # Filter to only numeric values
            probabilities = {k: float(v) for k, v in raw.items() if self._is_numeric(v)}
            if probabilities:
                return self._extract_from_probabilities(probabilities, source="raw_output")
        
        # No valid data found
        return 0.5, 0.5, {"error": "No probability data available"}
    
    def _extract_from_probabilities(
        self, 
        probabilities: Dict[str, float],
        source: str
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Extract confidence from a probabilities dict.
        
        Args:
            probabilities: {pathology: probability} dict
            source: Where the probabilities came from (for metadata)
            
        Returns:
            Tuple of (raw_confidence, uncertainty, metadata)
        """
        # Select target pathology or use max probability
        if self.target_pathology and self.target_pathology in probabilities:
            raw_conf = float(probabilities[self.target_pathology])
        else:
            raw_conf = max(probabilities.values()) if probabilities else 0.5
        
        uncertainty = 1.0 - raw_conf
        entropy = self._compute_entropy(probabilities)
        
        return raw_conf, uncertainty, {
            "extraction_method": "classification_precomputed" if source == "tool_confidence_data" else "classification_raw",
            "target_pathology": self.target_pathology,
            "all_probabilities": probabilities,
            "entropy": entropy,
            "source": source
        }
    
    @staticmethod
    def _is_numeric(value: Any) -> bool:
        """Check if value is numeric."""
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def _compute_entropy(probabilities: Dict[str, float]) -> float:
        """Compute normalized entropy from probability distribution."""
        probs = np.array([float(v) for v in probabilities.values()])
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        # Normalize by max entropy
        max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0


class SegmentationConfidenceExtractor(BaseConfidenceExtractor):
    """
    Confidence extractor for segmentation models (MedSAM, ChestXRaySegmentationTool).
    
    Expects tool to return:
        metadata["confidence_data"] = {
            "organ_name": {
                "mean_probability": float,
                "std_probability": float,
                ...
            }
        }
    
    Confidence = mean of per-organ mean_probability values
    Uncertainty = std across organs (or mean of per-organ std_probability)
    """
    
    @property
    def method_name(self) -> str:
        return "segmentation_precomputed"
    
    def extract(self, model_output: ModelOutput) -> Tuple[float, float, Dict[str, Any]]:
        """
        Extract confidence from segmentation output.
        
        Primary path: Uses pre-computed confidence_data from tool (per-organ stats).
        Fallback: Extracts from raw_output["metrics"] if confidence_data unavailable.
        """
        auxiliary = model_output.auxiliary
        raw = model_output.raw_output
        
        # Primary path: Use pre-computed confidence_data from tool
        confidence_data = auxiliary.get("confidence_data")
        if confidence_data and isinstance(confidence_data, dict):
            return self._extract_from_confidence_data(confidence_data)
        
        # Fallback: Extract from raw_output["metrics"] (tool's output structure)
        if isinstance(raw, dict) and "metrics" in raw:
            metrics = raw["metrics"]
            if isinstance(metrics, dict):
                return self._extract_from_metrics(metrics)
        
        # Legacy fallback: raw_output is directly {organ: {confidence_score: ...}}
        if isinstance(raw, dict):
            organ_confs = self._extract_organ_confidences_legacy(raw)
            if organ_confs:
                raw_conf = float(np.mean(organ_confs))
                uncertainty = float(np.std(organ_confs)) if len(organ_confs) > 1 else 1.0 - raw_conf
                return raw_conf, uncertainty, {
                    "extraction_method": "segmentation_legacy",
                    "num_organs": len(organ_confs),
                    "organ_confidences": organ_confs
                }
        
        # No valid data found
        return 0.5, 0.5, {"error": "No segmentation confidence data available"}
    
    def _extract_from_confidence_data(
        self, 
        confidence_data: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Extract confidence from tool's confidence_data structure.
        
        Expected format:
            {
                "Left Lung": {"mean_probability": 0.85, "std_probability": 0.1, ...},
                "Heart": {"mean_probability": 0.92, "std_probability": 0.05, ...}
            }
        """
        organ_confidences = []
        organ_uncertainties = []
        organ_details = {}
        
        for organ, data in confidence_data.items():
            if isinstance(data, dict) and "mean_probability" in data:
                mean_prob = float(data["mean_probability"])
                organ_confidences.append(mean_prob)
                organ_details[organ] = mean_prob
                
                if "std_probability" in data:
                    organ_uncertainties.append(float(data["std_probability"]))
        
        if not organ_confidences:
            return 0.5, 0.5, {"error": "No organ mean_probability found"}
        
        raw_conf = float(np.mean(organ_confidences))
        
        # Uncertainty: use std across organs, or mean of per-organ std
        if organ_uncertainties:
            uncertainty = float(np.mean(organ_uncertainties))
        else:
            uncertainty = float(np.std(organ_confidences)) if len(organ_confidences) > 1 else 1.0 - raw_conf
        
        return raw_conf, uncertainty, {
            "extraction_method": "segmentation_precomputed",
            "num_organs": len(organ_confidences),
            "organ_confidences": organ_details,
            "source": "tool_confidence_data"
        }
    
    def _extract_from_metrics(
        self, 
        metrics: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Extract confidence from raw_output["metrics"] structure.
        
        Expected format:
            {
                "Left Lung": {"confidence_score": 0.85, ...},
                "Heart": {"confidence_score": 0.92, ...}
            }
        """
        organ_confidences = []
        organ_details = {}
        
        for organ, data in metrics.items():
            if isinstance(data, dict):
                # Try different confidence field names
                conf = data.get("confidence_score") or data.get("mean_probability") or data.get("confidence")
                if conf is not None:
                    conf_float = float(conf)
                    organ_confidences.append(conf_float)
                    organ_details[organ] = conf_float
        
        if not organ_confidences:
            return 0.5, 0.5, {"error": "No organ confidence found in metrics"}
        
        raw_conf = float(np.mean(organ_confidences))
        uncertainty = float(np.std(organ_confidences)) if len(organ_confidences) > 1 else 1.0 - raw_conf
        
        return raw_conf, uncertainty, {
            "extraction_method": "segmentation_metrics",
            "num_organs": len(organ_confidences),
            "organ_confidences": organ_details,
            "source": "raw_output_metrics"
        }
    
    @staticmethod
    def _extract_organ_confidences_legacy(raw: Dict[str, Any]) -> List[float]:
        """
        Legacy extraction for older output formats.
        Looks for confidence_score or confidence in nested dicts.
        """
        organ_confs = []
        for key, value in raw.items():
            if isinstance(value, dict):
                conf = value.get("confidence_score") or value.get("confidence")
                if conf is not None:
                    try:
                        organ_confs.append(float(conf))
                    except (TypeError, ValueError):
                        pass
        return organ_confs


class VQAConfidenceExtractor(BaseConfidenceExtractor):
    """
    Confidence extractor for VQA models (CheXagent, LLaVA-Med).
    
    Expects tool to return:
        metadata["confidence_data"] = {
            "samples": List[str],              # Generated answer samples
            "consistency_score": float,        # Pre-computed consistency [0,1]
            "num_unique_answers": int,         # Number of unique answers
            "answer_distribution": Dict,       # Answer frequency distribution
            "consensus_answer": str            # Most common answer
        }
    
    Confidence is based on self-consistency across multiple generated answers.
    Reference: Wang et al. 2022 "Self-Consistency Improves Chain of Thought Reasoning"
    """
    
    @property
    def method_name(self) -> str:
        return "self_consistency"
    
    def extract(self, model_output: ModelOutput) -> Tuple[float, float, Dict[str, Any]]:
        """
        Extract confidence from VQA output.
        
        Primary path: Uses pre-computed consistency_score from tool.
        """
        auxiliary = model_output.auxiliary
        
        # Primary path: Use pre-computed confidence_data from VQA tool
        confidence_data = auxiliary.get("confidence_data")
        if confidence_data and isinstance(confidence_data, dict):
            if "consistency_score" in confidence_data:
                raw_conf = float(confidence_data["consistency_score"])
                uncertainty = 1.0 - raw_conf
                
                samples = confidence_data.get("samples", [])
                return raw_conf, uncertainty, {
                    "extraction_method": "vqa_self_consistency_precomputed",
                    "num_samples": len(samples),
                    "num_unique_answers": confidence_data.get("num_unique_answers"),
                    "consensus_answer": confidence_data.get("consensus_answer"),
                    "answer_distribution": confidence_data.get("answer_distribution"),
                    "agreement_ratio": raw_conf,
                    "source": "tool_confidence_data"
                }
        
        # No confidence data available
        return 0.5, 0.5, {
            "extraction_method": "vqa_fallback",
            "error": "No confidence_data from VQA tool"
        }


class GroundingConfidenceExtractor(BaseConfidenceExtractor):
    """
    Confidence extractor for grounding models (MAIRA-2, XRayPhraseGroundingTool).
    
    Expects tool to return:
        metadata["confidence_data"] = {
            "confidence_score": float,    # Primary confidence [0,1]
            "has_prediction": bool,       # Whether any boxes were found
            "num_boxes": int,             # Number of bounding boxes
            "coverage_ratio": float       # Total area coverage
        }
    
    Confidence is computed by the grounding tool based on:
    - Number of detected boxes
    - Box coverage ratio (penalizes very large/small boxes)
    - Detection consistency
    """
    
    @property
    def method_name(self) -> str:
        return "grounding_confidence"
    
    def extract(self, model_output: ModelOutput) -> Tuple[float, float, Dict[str, Any]]:
        """
        Extract confidence from grounding output.
        
        Primary path: Uses pre-computed confidence_score from tool.
        """
        auxiliary = model_output.auxiliary
        
        # Primary path: Use pre-computed confidence_data from grounding tool
        confidence_data = auxiliary.get("confidence_data")
        if confidence_data and isinstance(confidence_data, dict):
            if "confidence_score" in confidence_data:
                raw_conf = float(confidence_data["confidence_score"])
                uncertainty = 1.0 - raw_conf
                
                return raw_conf, uncertainty, {
                    "extraction_method": "grounding_precomputed",
                    "has_prediction": confidence_data.get("has_prediction", True),
                    "num_boxes": confidence_data.get("num_boxes", 0),
                    "coverage_ratio": confidence_data.get("coverage_ratio", 0),
                    "avg_coverage": confidence_data.get("avg_coverage"),
                    "source": "tool_confidence_data"
                }
        
        # No confidence data available
        return 0.5, 0.5, {
            "extraction_method": "grounding_fallback",
            "error": "No confidence_data from grounding tool"
        }


class ReportConfidenceExtractor(BaseConfidenceExtractor):
    """
    Confidence extractor for radiology report generation models (ViT-BERT).
    
    Expects tool to return:
        metadata["confidence_data"] = {
            "overall_confidence": float,     # Primary confidence [0,1]
            "findings": {
                "consistency_score": float,
                "exact_consistency": float,
                "avg_similarity": float,
                "samples": [...]
            },
            "impression": {
                "consistency_score": float,
                ...
            }
        }
    
    Confidence is based on self-consistency across multiple generated reports.
    """

    @property
    def method_name(self) -> str:
        return "report_self_consistency"

    def extract(self, model_output: ModelOutput) -> Tuple[float, float, Dict[str, Any]]:
        """
        Extract confidence from report generation output.
        
        Primary path: Uses pre-computed overall_confidence from tool.
        Fallback: Computes weighted average from section scores.
        """
        auxiliary = model_output.auxiliary
        
        # Primary path: Use pre-computed confidence_data from tool
        confidence_data = auxiliary.get("confidence_data")
        if confidence_data and isinstance(confidence_data, dict):
            # Best case: overall_confidence is already computed
            if "overall_confidence" in confidence_data:
                raw_conf = float(confidence_data["overall_confidence"])
                uncertainty = 1.0 - raw_conf
                
                return raw_conf, uncertainty, {
                    "extraction_method": "report_consistency_precomputed",
                    "overall_confidence": raw_conf,
                    "findings_confidence": confidence_data.get("findings", {}).get("consistency_score"),
                    "impression_confidence": confidence_data.get("impression", {}).get("consistency_score"),
                    "source": "tool_confidence_data"
                }
            
            # Fallback: Compute from section scores
            findings_score = confidence_data.get("findings", {}).get("consistency_score")
            impression_score = confidence_data.get("impression", {}).get("consistency_score")
            
            if findings_score is not None or impression_score is not None:
                if findings_score is not None and impression_score is not None:
                    raw_conf = 0.6 * findings_score + 0.4 * impression_score
                else:
                    raw_conf = findings_score if findings_score is not None else impression_score
                    
                uncertainty = 1.0 - raw_conf
                return raw_conf, uncertainty, {
                    "extraction_method": "report_section_consistency",
                    "findings_score": findings_score,
                    "impression_score": impression_score,
                    "source": "tool_confidence_data"
                }
        
        # No confidence data available
        return 0.5, 0.5, {
            "extraction_method": "report_fallback",
            "error": "No confidence_data from report generator tool"
        }


class GenerationConfidenceExtractor(BaseConfidenceExtractor):
    """
    Confidence extractor for image generation models (RoentGen/Stable Diffusion).
    
    Expects tool to return:
        metadata["confidence_data"] = {
            "consistency_score": float,      # Primary confidence [0,1]
            "avg_pixel_similarity": float,   # Raw correlation coefficient
            "std_similarity": float,         # Variance across samples
            "num_samples": int               # Number of images generated
        }
    
    Confidence is based on consistency across multiple generated images.
    Higher consistency = model is more "confident" about the visual representation.
    """
    
    @property
    def method_name(self) -> str:
        return "generation_consistency"
    
    def extract(self, model_output: ModelOutput) -> Tuple[float, float, Dict[str, Any]]:
        """
        Extract confidence from image generation output.
        
        Primary path: Uses pre-computed consistency_score from tool.
        """
        auxiliary = model_output.auxiliary
        
        # Primary path: Use pre-computed confidence_data from generator tool
        confidence_data = auxiliary.get("confidence_data")
        if confidence_data and isinstance(confidence_data, dict):
            if "consistency_score" in confidence_data:
                raw_conf = float(confidence_data["consistency_score"])
                uncertainty = 1.0 - raw_conf
                
                return raw_conf, uncertainty, {
                    "extraction_method": "generation_consistency_precomputed",
                    "consistency_score": raw_conf,
                    "avg_pixel_similarity": confidence_data.get("avg_pixel_similarity"),
                    "std_similarity": confidence_data.get("std_similarity"),
                    "num_samples": confidence_data.get("num_samples"),
                    "source": "tool_confidence_data"
                }
        
        # No confidence data available
        return 0.5, 0.5, {
            "extraction_method": "generation_fallback",
            "error": "No confidence_data from generator tool"
        }


class ConfidenceNormalizer:
    """
    Normalizes raw confidence scores to [0, 1] range.
    
    Uses min-max normalization with per-task or per-model bounds.
    """
    
    def __init__(self):
        """Initialize with default bounds per task type."""
        self.bounds: Dict[str, Dict[str, Tuple[float, float]]] = {
            # task_type -> model_name -> (min, max)
        }
        # Default bounds for each task type
        self.default_bounds: Dict[str, Tuple[float, float]] = {
            TaskType.CLASSIFICATION.value: (0.0, 1.0),
            TaskType.SEGMENTATION.value: (0.0, 1.0),
            TaskType.VQA.value: (0.1, 0.9),  # Text-based confidence rarely extreme
            TaskType.GROUNDING.value: (0.0, 1.0),
            TaskType.REPORT.value: (0.1, 0.9),
            TaskType.GENERATION.value: (0.0, 1.0),
            TaskType.UNKNOWN.value: (0.0, 1.0)
        }
    
    def set_bounds(
        self, 
        task_type: str, 
        model_name: str, 
        min_val: float, 
        max_val: float
    ) -> None:
        """
        Set normalization bounds for a specific task-model combination.
        
        Args:
            task_type: Type of task
            model_name: Name of the model
            min_val: Minimum expected confidence value
            max_val: Maximum expected confidence value
        """
        if task_type not in self.bounds:
            self.bounds[task_type] = {}
        self.bounds[task_type][model_name] = (min_val, max_val)
    
    def normalize(
        self, 
        confidence: float, 
        task_type: str, 
        model_name: str = "default"
    ) -> float:
        """
        Normalize confidence score to [0, 1].
        
        Args:
            confidence: Raw confidence score
            task_type: Type of task
            model_name: Name of the model
            
        Returns:
            Normalized confidence ∈ [0, 1]
        """
        # Get bounds
        if task_type in self.bounds and model_name in self.bounds[task_type]:
            min_val, max_val = self.bounds[task_type][model_name]
        elif task_type in self.default_bounds:
            min_val, max_val = self.default_bounds[task_type]
        else:
            min_val, max_val = 0.0, 1.0
        
        # Min-max normalization
        range_val = max_val - min_val
        if range_val <= 0:
            return confidence
        
        normalized = (confidence - min_val) / (range_val + 1e-8)
        return max(0.0, min(1.0, normalized))


class ConfidenceCalibrator:
    """
    Calibrates confidence scores using isotonic regression or temperature scaling.
    
    Calibration ensures that confidence scores reflect true probabilities:
    - A confidence of 0.8 should mean 80% of predictions with that confidence are correct
    """
    
    def __init__(self, method: str = "isotonic"):
        """
        Initialize the calibrator.
        
        Args:
            method: Calibration method ("isotonic", "temperature", "platt")
        """
        self.method = method
        self.calibrators: Dict[str, Any] = {}  # task_model_key -> calibrator
        self.temperatures: Dict[str, float] = {}  # For temperature scaling
        self.is_fitted: Dict[str, bool] = {}
    
    def fit(
        self, 
        confidences: np.ndarray, 
        labels: np.ndarray,
        task_type: str,
        model_name: str = "default"
    ) -> None:
        """
        Fit the calibrator on validation data.
        
        Args:
            confidences: Array of confidence scores
            labels: Array of correctness labels (0 or 1)
            task_type: Type of task
            model_name: Name of the model
        """
        key = f"{task_type}_{model_name}"
        
        if not SKLEARN_AVAILABLE:
            warnings.warn("sklearn not available, calibration disabled")
            self.is_fitted[key] = False
            return
        
        confidences = np.array(confidences).flatten()
        labels = np.array(labels).flatten()
        
        if len(confidences) < 2:
            warnings.warn(f"Not enough samples for calibration: {len(confidences)}")
            self.is_fitted[key] = False
            return
        
        if self.method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(confidences, labels)
            self.calibrators[key] = calibrator
        
        elif self.method == "temperature":
            # Temperature scaling for classification
            temperature = self._fit_temperature(confidences, labels)
            self.temperatures[key] = temperature
        
        elif self.method == "platt":
            # Platt scaling using logistic regression
            X = confidences.reshape(-1, 1)
            calibrator = LogisticRegression()
            calibrator.fit(X, labels)
            self.calibrators[key] = calibrator
        
        self.is_fitted[key] = True
    
    def calibrate(
        self, 
        confidence: float, 
        task_type: str, 
        model_name: str = "default"
    ) -> Tuple[float, bool]:
        """
        Calibrate a confidence score.
        
        Args:
            confidence: Raw (normalized) confidence score
            task_type: Type of task
            model_name: Name of the model
            
        Returns:
            Tuple of (calibrated_confidence, was_calibrated)
        """
        key = f"{task_type}_{model_name}"
        
        if key not in self.is_fitted or not self.is_fitted[key]:
            return confidence, False
        
        if self.method == "isotonic":
            if key in self.calibrators:
                calibrated = self.calibrators[key].predict([confidence])[0]
                return float(calibrated), True
        
        elif self.method == "temperature":
            if key in self.temperatures:
                # Apply temperature scaling
                temp = self.temperatures[key]
                logit = np.log(confidence / (1 - confidence + 1e-10) + 1e-10)
                scaled_logit = logit / temp
                calibrated = 1.0 / (1.0 + np.exp(-scaled_logit))
                return float(calibrated), True
        
        elif self.method == "platt":
            if key in self.calibrators:
                calibrated = self.calibrators[key].predict_proba([[confidence]])[0, 1]
                return float(calibrated), True
        
        return confidence, False
    
    def _fit_temperature(
        self, 
        confidences: np.ndarray, 
        labels: np.ndarray
    ) -> float:
        """Fit optimal temperature using NLL minimization."""
        from scipy.optimize import minimize_scalar
        
        def nll_loss(temperature):
            # Apply temperature scaling
            logits = np.log(confidences / (1 - confidences + 1e-10) + 1e-10)
            scaled_logits = logits / temperature
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            
            # Compute NLL
            nll = -np.mean(
                labels * np.log(probs + 1e-10) + 
                (1 - labels) * np.log(1 - probs + 1e-10)
            )
            return nll
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        return result.x
    
    def save(self, path: Path) -> None:
        """Save calibrators to disk."""
        data = {
            "method": self.method,
            "calibrators": self.calibrators,
            "temperatures": self.temperatures,
            "is_fitted": self.is_fitted
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, path: Path) -> None:
        """Load calibrators from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.method = data["method"]
        self.calibrators = data["calibrators"]
        self.temperatures = data["temperatures"]
        self.is_fitted = data["is_fitted"]


class ConfidenceScoringPipeline:
    """
    Main pipeline that orchestrates confidence extraction, normalization, and calibration.
    
    Usage:
        pipeline = ConfidenceScoringPipeline()
        
        # Process model output
        model_output = ModelOutput(
            task_type="classification",
            raw_output={"Pneumonia": 0.85, "Cardiomegaly": 0.23},
            model_name="chest_xray_classifier"
        )
        
        result = pipeline.process(model_output)
        print(result.calibrated_confidence)  # 0.78 (calibrated)
    """
    
    def __init__(
        self, 
        calibration_method: str = "isotonic",
        calibrators_path: Optional[Path] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            calibration_method: Method for calibration ("isotonic", "temperature", "platt")
            calibrators_path: Path to load pre-trained calibrators from
        """
        # Initialize extractors for each task type
        self.extractors: Dict[str, BaseConfidenceExtractor] = {
            TaskType.CLASSIFICATION.value: ClassificationConfidenceExtractor(),
            TaskType.SEGMENTATION.value: SegmentationConfidenceExtractor(),
            TaskType.VQA.value: VQAConfidenceExtractor(),
            TaskType.GROUNDING.value: GroundingConfidenceExtractor(),
            TaskType.REPORT.value: ReportConfidenceExtractor(),
            TaskType.GENERATION.value: GenerationConfidenceExtractor(),
        }
        
        self.normalizer = ConfidenceNormalizer()
        self.calibrator = ConfidenceCalibrator(method=calibration_method)
        
        if calibrators_path and calibrators_path.exists():
            self.calibrator.load(calibrators_path)
    
    def register_extractor(
        self, 
        task_type: str, 
        extractor: BaseConfidenceExtractor
    ) -> None:
        """
        Register a custom extractor for a task type.
        
        Args:
            task_type: Type of task
            extractor: Custom extractor instance
        """
        self.extractors[task_type] = extractor
    
    def process(
        self, 
        model_output: ModelOutput,
        target_pathology: Optional[str] = None
    ) -> ConfidenceResult:
        """
        Process a model output and return calibrated confidence result.
        
        Args:
            model_output: Standardized model output
            target_pathology: Specific pathology to extract confidence for (optional)
            
        Returns:
            ConfidenceResult with calibrated confidence
        """
        task_type = model_output.task_type
        model_name = model_output.model_name
        
        # Step 1: Extract raw confidence
        extractor = self._get_extractor(task_type, target_pathology)
        raw_conf, uncertainty, extraction_meta = extractor.extract(model_output)
        
        # Step 2: Normalize
        normalized_conf = self.normalizer.normalize(raw_conf, task_type, model_name)
        
        # Step 3: Calibrate
        calibrated_conf, was_calibrated = self.calibrator.calibrate(
            normalized_conf, task_type, model_name
        )
        
        # Build result
        method_parts = [extractor.method_name]
        if was_calibrated:
            method_parts.append(self.calibrator.method)
        
        result = ConfidenceResult(
            task_type=task_type,
            raw_confidence=raw_conf,
            uncertainty=uncertainty,
            calibrated_confidence=calibrated_conf,
            method=" + ".join(method_parts),
            model_name=model_name,
            metadata={
                "extraction": extraction_meta,
                "normalized_confidence": normalized_conf,
                "calibration_applied": was_calibrated
            },
            is_calibrated=was_calibrated
        )
        
        return result
    
    def process_batch(
        self, 
        model_outputs: List[ModelOutput]
    ) -> List[ConfidenceResult]:
        """
        Process multiple model outputs.
        
        Args:
            model_outputs: List of model outputs
            
        Returns:
            List of confidence results
        """
        return [self.process(output) for output in model_outputs]
    
    def fit_calibrators(
        self, 
        calibration_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """
        Fit calibrators using validation data.
        
        Args:
            calibration_data: Dict mapping "task_model" keys to (confidences, labels) tuples
        """
        for key, (confidences, labels) in calibration_data.items():
            parts = key.split("_", 1)
            task_type = parts[0]
            model_name = parts[1] if len(parts) > 1 else "default"
            self.calibrator.fit(confidences, labels, task_type, model_name)
    
    def _get_extractor(
        self, 
        task_type: str,
        target_pathology: Optional[str] = None
    ) -> BaseConfidenceExtractor:
        """Get the appropriate extractor for a task type."""
        if task_type == TaskType.CLASSIFICATION.value and target_pathology:
            return ClassificationConfidenceExtractor(target_pathology=target_pathology)
        
        if task_type in self.extractors:
            return self.extractors[task_type]
        
        # Fallback to VQA extractor (keyword analysis)
        return VQAConfidenceExtractor(use_self_consistency=False)


class ConfidenceFusion:
    """
    Fuses confidence scores from multiple models/tools.
    
    Methods:
    - Weighted average: final_conf = Σ(w_i * conf_i)
    - Learned weights: weights from logistic regression
    """
    
    def __init__(self, fusion_method: str = "weighted_average"):
        """
        Initialize the fusion module.
        
        Args:
            fusion_method: Method for fusion ("weighted_average", "learned")
        """
        self.fusion_method = fusion_method
        self.weights: Dict[str, float] = {}  # model_name -> weight
        self.learned_model: Optional[Any] = None
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set manual weights for each model.
        
        Args:
            weights: Dict mapping model names to weights (should sum to 1)
        """
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
    
    def fit(
        self, 
        confidences: np.ndarray, 
        labels: np.ndarray,
        model_names: List[str]
    ) -> None:
        """
        Learn optimal fusion weights using logistic regression.
        
        Args:
            confidences: Array of shape (n_samples, n_models)
            labels: Array of correctness labels
            model_names: Names of models corresponding to columns
        """
        if not SKLEARN_AVAILABLE:
            warnings.warn("sklearn not available, using equal weights")
            self.weights = {name: 1.0 / len(model_names) for name in model_names}
            return
        
        # Fit logistic regression to learn weights
        lr = LogisticRegression(fit_intercept=False, max_iter=1000)
        lr.fit(confidences, labels)
        
        # Extract coefficients as weights
        coefs = lr.coef_[0]
        # Softmax to get positive weights that sum to 1
        exp_coefs = np.exp(coefs - np.max(coefs))
        normalized = exp_coefs / exp_coefs.sum()
        
        self.weights = {name: float(w) for name, w in zip(model_names, normalized)}
        self.learned_model = lr
    
    def fuse(self, results: List[ConfidenceResult]) -> float:
        """
        Fuse confidence scores from multiple results.
        
        Args:
            results: List of confidence results from different models
            
        Returns:
            Fused confidence score
        """
        if not results:
            return 0.5
        
        if self.fusion_method == "weighted_average":
            # Use model-specific weights or equal weights
            total_weight = 0.0
            weighted_sum = 0.0
            
            for r in results:
                weight = self.weights.get(r.model_name, 1.0 / len(results))
                weighted_sum += weight * r.calibrated_confidence
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.5
        
        elif self.fusion_method == "learned" and self.learned_model is not None:
            # Use learned model
            confs = np.array([[r.calibrated_confidence for r in results]])
            return float(self.learned_model.predict_proba(confs)[0, 1])
        
        else:
            # Simple average fallback
            return float(np.mean([r.calibrated_confidence for r in results]))


class CalibrationMetrics:
    """
    Computes validation metrics for confidence calibration.
    
    Metrics:
    - Expected Calibration Error (ECE)
    - Brier Score
    - AUROC vs correctness
    """
    
    @staticmethod
    def expected_calibration_error(
        confidences: np.ndarray, 
        labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        ECE = Σ |bin_accuracy - bin_confidence| * bin_size / total_samples
        
        Args:
            confidences: Array of confidence scores
            labels: Array of correctness labels (0 or 1)
            n_bins: Number of bins
            
        Returns:
            ECE score (lower is better, 0 is perfect)
        """
        confidences = np.array(confidences)
        labels = np.array(labels)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = labels[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
        
        return float(ece)
    
    @staticmethod
    def brier_score(confidences: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Brier Score.
        
        BS = mean((confidence - label)^2)
        
        Args:
            confidences: Array of confidence scores
            labels: Array of correctness labels (0 or 1)
            
        Returns:
            Brier score (lower is better, 0 is perfect)
        """
        if SKLEARN_AVAILABLE:
            return float(brier_score_loss(labels, confidences))
        
        confidences = np.array(confidences)
        labels = np.array(labels)
        return float(np.mean((confidences - labels) ** 2))
    
    @staticmethod
    def auroc(confidences: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute AUROC (Area Under ROC Curve).
        
        Args:
            confidences: Array of confidence scores
            labels: Array of correctness labels (0 or 1)
            
        Returns:
            AUROC score (higher is better, 1 is perfect)
        """
        if not SKLEARN_AVAILABLE:
            warnings.warn("sklearn not available, cannot compute AUROC")
            return 0.5
        
        try:
            return float(roc_auc_score(labels, confidences))
        except ValueError:
            # Only one class present
            return 0.5
    
    @staticmethod
    def reliability_diagram(
        confidences: np.ndarray, 
        labels: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, List[float]]:
        """
        Compute data for reliability diagram.
        
        Args:
            confidences: Array of confidence scores
            labels: Array of correctness labels
            n_bins: Number of bins
            
        Returns:
            Dict with 'bin_midpoints', 'bin_accuracies', 'bin_confidences', 'bin_counts'
        """
        confidences = np.array(confidences)
        labels = np.array(labels)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        midpoints = []
        accuracies = []
        avg_confidences = []
        counts = []
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            count = in_bin.sum()
            
            midpoints.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            
            if count > 0:
                accuracies.append(float(labels[in_bin].mean()))
                avg_confidences.append(float(confidences[in_bin].mean()))
                counts.append(int(count))
            else:
                accuracies.append(0.0)
                avg_confidences.append(0.0)
                counts.append(0)
        
        return {
            "bin_midpoints": midpoints,
            "bin_accuracies": accuracies,
            "bin_confidences": avg_confidences,
            "bin_counts": counts
        }


class ConfidenceLogger:
    """
    Logs confidence outputs with full traceability.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir or Path("logs/confidence")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logs: List[Dict[str, Any]] = []
    
    def log(
        self, 
        result: ConfidenceResult,
        dataset: str = "unknown",
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a confidence result.
        
        Args:
            result: Confidence result to log
            dataset: Name of the dataset
            additional_info: Additional context
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": result.model_name,
            "task": result.task_type,
            "confidence": result.calibrated_confidence,
            "raw_confidence": result.raw_confidence,
            "uncertainty": result.uncertainty,
            "method": result.method,
            "calibrated": result.is_calibrated,
            "dataset": dataset,
            "confidence_bin": self._get_confidence_bin(result.calibrated_confidence),
            "additional_info": additional_info or {}
        }
        
        self.logs.append(log_entry)
    
    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save logs to file.
        
        Args:
            filename: Optional filename (default: timestamp-based)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"confidence_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(self.logs, f, indent=2)
        
        return filepath
    
    @staticmethod
    def _get_confidence_bin(confidence: float) -> str:
        """Get confidence histogram bin."""
        bins = [(0, 0.2, "very_low"), (0.2, 0.4, "low"), (0.4, 0.6, "medium"),
                (0.6, 0.8, "high"), (0.8, 1.0, "very_high")]
        
        for low, high, name in bins:
            if low <= confidence < high:
                return name
        return "very_high"  # For confidence == 1.0


# Convenience function for quick confidence scoring
def score_confidence(
    raw_output: Any,
    task_type: str,
    model_name: str = "unknown",
    auxiliary: Optional[Dict[str, Any]] = None,
    target_pathology: Optional[str] = None
) -> ConfidenceResult:
    """
    Convenience function to quickly score confidence from any model output.
    
    Args:
        raw_output: Raw model output
        task_type: Type of task ("classification", "segmentation", etc.)
        model_name: Name of the model
        auxiliary: Additional data (logits, attention, etc.)
        target_pathology: Specific pathology to extract confidence for
        
    Returns:
        ConfidenceResult with calibrated confidence
    """
    pipeline = ConfidenceScoringPipeline()
    
    model_output = ModelOutput(
        task_type=task_type,
        raw_output=raw_output,
        auxiliary=auxiliary or {},
        model_name=model_name
    )
    
    return pipeline.process(model_output, target_pathology=target_pathology)
