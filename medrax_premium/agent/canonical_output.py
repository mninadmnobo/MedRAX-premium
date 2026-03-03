"""
Canonical output format for all MedRAX tools.
Converts heterogeneous tool outputs into a unified, comparable format.

Now integrated with the unified confidence scoring pipeline for
model-agnostic, calibrated confidence scores.
"""

from dataclasses import dataclass, asdict
from typing import Any, Optional, List, Dict
from datetime import datetime
import json

# Import the unified confidence scoring pipeline
from .confidence_scoring import (
    ModelOutput,
    ConfidenceResult,
    ConfidenceScoringPipeline,
    TaskType,
    score_confidence
)

# Global pipeline instance (lazy initialization)
_confidence_pipeline: Optional[ConfidenceScoringPipeline] = None


def get_confidence_pipeline() -> ConfidenceScoringPipeline:
    """Get or create the global confidence scoring pipeline."""
    global _confidence_pipeline
    if _confidence_pipeline is None:
        _confidence_pipeline = ConfidenceScoringPipeline(calibration_method="isotonic")
    return _confidence_pipeline


@dataclass
class CanonicalFinding:
    """
    Unified representation for all tool outputs.
    
    This makes outputs from different tools (segmentation, classification, VQA)
    comparable by converting them into a common schema.
    
    Attributes:
        pathology: Disease/finding name (e.g., "Pneumothorax", "Cardiomegaly")
        region: Anatomical location (e.g., "right upper lobe", "bilateral", "global")
        confidence: Calibrated confidence score (0.0 to 1.0)
        evidence_type: Type of evidence ("classification", "segmentation", "vqa", "report", "grounding")
        source_tool: Which tool produced this finding
        raw_value: Original output from the tool
        metadata: Additional tool-specific information
    """
    pathology: str
    region: str
    confidence: float
    evidence_type: str
    source_tool: str
    raw_value: Any
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# Confidence score
# These are learned from validation data to make confidence scores honest
# TODO: Tune these values on your validation set
CALIBRATION_PARAMS = {
    "chest_xray_classifier": {
        "scale": 0.87,  # Classifier tends to be overconfident
        "description": "DenseNet-121 classification model"
    },
    "chest_xray_expert": {
        "scale": 1.05,  # VQA tends to be slightly underconfident
        "description": "CheXagent VQA model"
    },
    "segmentation_tool": {
        "scale": 0.92,  # Segmentation confidence not well-calibrated
        "description": "MedSAM segmentation model"
    },
    "chest_xray_report_generator": {
        "scale": 0.95,  # Report generator fairly well-calibrated
        "description": "Report generation model"
    },
    "phrase_grounding_tool": {
        "scale": 0.90,  # Grounding scores need adjustment
        "description": "Phrase grounding model"
    },
    "llava_med": {
        "scale": 1.0,  # LLaVA-Med baseline
        "description": "LLaVA-Med VQA model"
    }
}


def calibrate_confidence(tool_name: str, raw_confidence: float) -> float:
    """
    Calibrate confidence scores to make them comparable across tools.
    
    Different models have different confidence distributions. A 0.9 from one model
    might be more reliable than a 0.9 from another. This function normalizes them.
    
    Args:
        tool_name: Name of the tool
        raw_confidence: Original confidence score (0-1)
        
    Returns:
        Calibrated confidence score (0-1)
    """
    if tool_name not in CALIBRATION_PARAMS:
        return raw_confidence  # No calibration available, use raw
    
    scale = CALIBRATION_PARAMS[tool_name]["scale"]
    calibrated = raw_confidence * scale
    
    # Ensure bounds [0, 1]
    return max(0.0, min(1.0, calibrated))


def estimate_text_confidence(text: str) -> float:
    """
    Estimate confidence from text-based outputs (VQA, reports).
    
    Uses keyword analysis to infer confidence levels from natural language.
    
    Args:
        text: Natural language output
        
    Returns:
        Estimated confidence (0-1)
    """
    text_lower = text.lower()
    
    # High confidence indicators (0.85)
    high_conf_words = [
        "definitely", "clearly", "obvious", "marked", "severe", 
        "significant", "prominent", "extensive"
    ]
    
    # Medium confidence indicators (0.65)
    med_conf_words = [
        "likely", "probable", "appears", "suggests", "moderate", 
        "mild", "consistent with"
    ]
    
    # Low confidence indicators (0.45)
    low_conf_words = [
        "possible", "questionable", "subtle", "uncertain", "may be",
        "could be", "perhaps"
    ]
    
    # Negative indicators (0.1 - finding is absent)
    neg_words = [
        "no evidence", "not seen", "absent", "unremarkable", "normal",
        "no significant", "without"
    ]
    
    # Count indicators
    high_count = sum(1 for word in high_conf_words if word in text_lower)
    med_count = sum(1 for word in med_conf_words if word in text_lower)
    low_count = sum(1 for word in low_conf_words if word in text_lower)
    neg_count = sum(1 for word in neg_words if word in text_lower)
    
    # Priority: negative > high > medium > low
    if neg_count > 0:
        return 0.1  # Finding is explicitly absent
    elif high_count > 0:
        return 0.85
    elif med_count > 0:
        return 0.65
    elif low_count > 0:
        return 0.45
    else:
        return 0.5  # Neutral/unclear


def normalize_classification_output(
    output: Dict[str, Any], 
    tool_name: str,
    min_confidence_threshold: float = 0.01
) -> List[CanonicalFinding]:
    """
    Convert classification tool output to canonical format using unified confidence scoring.
    
    Uses the ConfidenceScoringPipeline for model-agnostic confidence extraction,
    normalization, and calibration.
    
    Args:
        output: Raw output from classifier (dict of pathology: probability or tuple)
        tool_name: Name of the classification tool
        min_confidence_threshold: Minimum confidence to include a finding (default: 0.01)
        
    Returns:
        List of CanonicalFinding objects with calibrated confidence scores
    """
    findings = []
    pipeline = get_confidence_pipeline()
    
    # Extract the actual output dict (handle tuple format)
    if isinstance(output, tuple):
        output_dict = output[0] if output and len(output) > 0 else {}
        metadata = output[1] if len(output) > 1 else {}
    elif isinstance(output, dict):
        output_dict = output
        metadata = {}
    else:
        return findings
    
    if not isinstance(output_dict, dict):
        return findings
    
    for pathology, prob in output_dict.items():
        try:
            prob_float = float(prob)
        except (TypeError, ValueError):
            continue
        
        if prob_float < min_confidence_threshold:
            continue
        
        # Build auxiliary data for confidence scoring
        auxiliary_data = {"logits": None}
        if "confidence_data" in metadata:
            auxiliary_data["confidence_data"] = metadata["confidence_data"]
        
        # Use unified confidence scoring pipeline
        try:
            model_output = ModelOutput(
                task_type=TaskType.CLASSIFICATION.value,
                raw_output={pathology: prob_float},
                auxiliary=auxiliary_data,
                model_name=tool_name
            )
            
            confidence_result = pipeline.process(model_output, target_pathology=pathology)
            calibrated_conf = confidence_result.calibrated_confidence
            uncertainty = confidence_result.uncertainty
        except Exception as e:
            calibrated_conf = calibrate_confidence(tool_name, prob_float)
            uncertainty = 1.0 - calibrated_conf
        
        try:
            finding = CanonicalFinding(
                pathology=pathology,
                region="global",
                confidence=calibrated_conf,
                evidence_type="classification",
                source_tool=tool_name,
                raw_value=prob_float,
                metadata={
                    "raw_probability": prob_float,
                    "calibration_applied": True,
                    "uncertainty": uncertainty,
                    "confidence_method": confidence_result.method if 'confidence_result' in dir() else "legacy",
                    **metadata
                }
            )
            findings.append(finding)
        except Exception:
            pass
    
    return findings


def normalize_vqa_output(
    output: Any, 
    tool_name: str, 
    prompt: str = "",
    samples: Optional[List[str]] = None
) -> List[CanonicalFinding]:
    """
    Convert VQA tool output to canonical format using unified confidence scoring.
    
    Uses the ConfidenceScoringPipeline for self-consistency confidence extraction.
    Now reads confidence_data from tool metadata for pre-computed scores.
    Expanded to handle negation ("no cardiomegaly" → low confidence) and to
    always return at least one finding so conflict detection can function.
    
    Args:
        output: Raw text output from VQA model (can be dict, tuple, or string)
        tool_name: Name of the VQA tool
        prompt: Original prompt/question
        samples: Optional multiple samples for self-consistency scoring
        
    Returns:
        List of CanonicalFinding objects with calibrated confidence scores
    """
    try:
        return _normalize_vqa_output_inner(output, tool_name, prompt, samples)
    except Exception as exc:
        # Absolute safety net — NEVER return [] for VQA; always give
        # conflict detection something to work with.
        text = ""
        if isinstance(output, tuple) and output:
            first = output[0]
            if isinstance(first, dict):
                text = first.get("response", first.get("error", str(first)))
            else:
                text = str(first)
        elif isinstance(output, dict):
            text = output.get("response", output.get("error", str(output)))
        else:
            text = str(output)
        return [CanonicalFinding(
            pathology="general_assessment",
            region="unspecified",
            confidence=0.5,
            evidence_type="vqa",
            source_tool=tool_name,
            raw_value=text[:500],
            metadata={"prompt": prompt, "fallback": True, "error": str(exc)}
        )]


def _normalize_vqa_output_inner(
    output: Any,
    tool_name: str,
    prompt: str = "",
    samples: Optional[List[str]] = None
) -> List[CanonicalFinding]:
    """Core VQA normalization (called by the public wrapper)."""
    pipeline = get_confidence_pipeline()
    
    # Handle tuple format (response_dict, metadata)
    auxiliary_data = {}
    metadata_dict = {}
    if isinstance(output, tuple):
        if len(output) > 1 and isinstance(output[1], dict):
            metadata_dict = output[1]
            # Extract confidence_data from tool metadata
            if "confidence_data" in metadata_dict:
                auxiliary_data["confidence_data"] = metadata_dict["confidence_data"]
        output = output[0] if output and len(output) > 0 else {}
    
    # Handle dict with error — still produce a low-confidence finding so that
    # conflict detection sees *something* from this tool
    if isinstance(output, dict) and "error" in output:
        return [CanonicalFinding(
            pathology="general_assessment",
            region="unspecified",
            confidence=0.1,
            evidence_type="vqa",
            source_tool=tool_name,
            raw_value=output.get("error", ""),
            metadata={"prompt": prompt, "tool_error": True}
        )]
    
    # Extract text
    if isinstance(output, dict):
        text = output.get("response", str(output))
    else:
        text = str(output)
    
    # Add samples for self-consistency if provided (override if not in confidence_data)
    if samples and "confidence_data" not in auxiliary_data:
        auxiliary_data["samples"] = samples
    
    # Use unified confidence scoring pipeline
    try:
        model_output = ModelOutput(
            task_type=TaskType.VQA.value,
            raw_output=text,
            auxiliary=auxiliary_data,
            model_name=tool_name
        )
        
        confidence_result = pipeline.process(model_output)
        calibrated_conf = confidence_result.calibrated_confidence
        uncertainty = confidence_result.uncertainty
        confidence_method = confidence_result.method
        
    except Exception as e:
        raw_confidence = estimate_text_confidence(text)
        calibrated_conf = calibrate_confidence(tool_name, raw_confidence)
        uncertainty = 1.0 - calibrated_conf
        confidence_method = "keyword_analysis_legacy"
    
    # ── Polarity-aware pathology extraction ──
    # Uses expanded synonym lists and checks for preceding negation phrases
    # so "no cardiomegaly" → (Cardiomegaly, False) → low confidence.
    polarity_results = extract_pathologies_with_polarity(text)
    
    findings = []
    if polarity_results:
        for pathology, is_positive in polarity_results:
            # Adjust per-pathology confidence based on polarity
            if is_positive:
                path_conf = calibrated_conf
            else:
                # Negative mention → flip confidence to reflect *absence*
                path_conf = max(0.05, 1.0 - calibrated_conf) if calibrated_conf > 0.5 else 0.1
            
            finding = CanonicalFinding(
                pathology=pathology,
                region="unspecified",
                confidence=path_conf,
                evidence_type="vqa",
                source_tool=tool_name,
                raw_value=text,
                metadata={
                    "prompt": prompt,
                    "full_response": text[:500],
                    "confidence_estimation_method": confidence_method,
                    "uncertainty": uncertainty,
                    "is_positive_mention": is_positive,
                    "self_consistency_samples": len(auxiliary_data.get("confidence_data", {}).get("samples", [])) or len(auxiliary_data.get("samples", []))
                }
            )
            findings.append(finding)
    else:
        # No specific pathology detected, create general finding
        finding = CanonicalFinding(
            pathology="general_assessment",
            region="unspecified",
            confidence=calibrated_conf,
            evidence_type="vqa",
            source_tool=tool_name,
            raw_value=text,
            metadata={
                "prompt": prompt,
                "full_response": text[:500],
                "confidence_estimation_method": confidence_method,
                "uncertainty": uncertainty
            }
        )
        findings.append(finding)
    
    return findings


def normalize_segmentation_output(
    output: Any, 
    tool_name: str,
    mask_probabilities: Optional[Any] = None
) -> List[CanonicalFinding]:
    """
    Convert segmentation tool output to canonical format using unified confidence scoring.
    
    Uses the ConfidenceScoringPipeline for mask probability-based confidence extraction.
    Now reads confidence_data from tool metadata for pre-computed scores.
    
    Args:
        output: Raw segmentation output (mask or dict) or tuple (output_dict, metadata)
        tool_name: Name of the segmentation tool
        mask_probabilities: Optional raw mask probabilities for confidence calculation
        
    Returns:
        List of CanonicalFinding objects with calibrated confidence scores
    """
    findings = []
    pipeline = get_confidence_pipeline()
    
    # Handle tuple format (output_dict, metadata)
    metadata_dict = {}
    if isinstance(output, tuple):
        if len(output) > 1 and isinstance(output[1], dict):
            metadata_dict = output[1]
        output = output[0] if output and len(output) > 0 else {}
    
    # Extract confidence_data from metadata if available
    confidence_data = metadata_dict.get("confidence_data", {})
    
    # Handle different segmentation output formats
    if isinstance(output, dict):
        # Get metrics dict if present
        metrics = output.get("metrics", output)
        
        for region, mask_info in metrics.items():
            if isinstance(mask_info, dict):
                area_pct = mask_info.get("area_percentage", 0.0)
                raw_confidence = mask_info.get("confidence", mask_info.get("confidence_score", 0.5))
                
                # Use pre-computed confidence from confidence_data if available
                if confidence_data and region in confidence_data:
                    region_conf = confidence_data[region]
                    raw_confidence = region_conf.get("mean_probability", raw_confidence)
            else:
                area_pct = 0.0
                raw_confidence = 0.5
            
            # Use unified confidence scoring pipeline
            try:
                model_output = ModelOutput(
                    task_type=TaskType.SEGMENTATION.value,
                    raw_output={region: mask_info},
                    auxiliary={
                        "confidence_score": raw_confidence,
                        "mask_probabilities": mask_probabilities,
                        "confidence_data": confidence_data.get(region, {}) if confidence_data else {}
                    },
                    model_name=tool_name
                )
                
                confidence_result = pipeline.process(model_output)
                calibrated_conf = confidence_result.calibrated_confidence
                uncertainty = confidence_result.uncertainty
                confidence_method = confidence_result.method
                
            except Exception as e:
                calibrated_conf = calibrate_confidence(tool_name, raw_confidence)
                uncertainty = 1.0 - calibrated_conf
                confidence_method = "legacy"
            
            finding = CanonicalFinding(
                pathology="segmented_region",
                region=region,
                confidence=calibrated_conf,
                evidence_type="segmentation",
                source_tool=tool_name,
                raw_value=mask_info,
                metadata={
                    "area_percentage": area_pct,
                    "raw_confidence": raw_confidence,
                    "uncertainty": uncertainty,
                    "confidence_method": confidence_method,
                    "mean_probability": confidence_data.get(region, {}).get("mean_probability") if confidence_data else None,
                    "high_confidence_ratio": confidence_data.get(region, {}).get("high_confidence_ratio") if confidence_data else None
                }
            )
            findings.append(finding)
    
    return findings


# Negation window — number of characters before a pathology mention to scan for negation
_NEGATION_WINDOW = 40
_NEGATION_PHRASES = [
    "no ", "no evidence", "not seen", "absent", "without ", "negative for",
    "ruled out", "unremarkable", "unlikely", "not identified", "not detected",
    "no significant", "no definite", "not consistent", "no obvious",
    "denies ", "no acute",
]

# Expanded synonyms / aliases keyed by the canonical DenseNet-121 label
_PATHOLOGY_ALIASES: Dict[str, List[str]] = {
    "Atelectasis":       ["atelectasis", "atelectatic", "volume loss",
                          "collapsed lobe", "subsegmental collapse"],
    "Cardiomegaly":      ["cardiomegaly", "cardiac enlargement", "enlarged heart",
                          "enlarged cardiac silhouette", "heart is enlarged",
                          "cardiac silhouette is enlarged", "heart enlargement",
                          "heart size is enlarged"],
    "Consolidation":     ["consolidation", "consolidated", "airspace disease",
                          "air-space disease", "airspace opacity",
                          "lobar consolidation"],
    "Edema":             ["edema", "oedema", "pulmonary edema",
                          "fluid overload", "vascular congestion",
                          "interstitial edema", "alveolar edema",
                          "pulmonary congestion"],
    "Effusion":          ["effusion", "pleural effusion", "pleural fluid",
                          "fluid in the pleural", "costophrenic blunting",
                          "blunted costophrenic", "meniscus sign"],
    "Emphysema":         ["emphysema", "hyperinflation", "hyperinflated",
                          "copd", "over-inflation", "barrel chest",
                          "flattened diaphragm"],
    "Fibrosis":          ["fibrosis", "fibrotic", "scarring", "pulmonary fibrosis",
                          "interstitial fibrosis", "reticular pattern"],
    "Fracture":          ["fracture", "fractured", "broken rib", "rib fracture",
                          "cortical disruption"],
    "Hernia":            ["hernia", "hiatal hernia", "diaphragmatic hernia",
                          "hiatus hernia"],
    "Infiltration":      ["infiltration", "infiltrate", "infiltrates",
                          "opacity", "opacities", "opacification",
                          "hazy opacity", "ground-glass", "ground glass",
                          "lung opacity"],
    "Mass":              ["mass", "masses", "tumor", "tumour", "neoplasm",
                          "lung mass", "mediastinal mass"],
    "Nodule":            ["nodule", "nodular", "nodules",
                          "pulmonary nodule", "lung nodule",
                          "solitary pulmonary nodule"],
    "Pleural Thickening":["pleural thickening", "thickened pleura",
                          "pleural thickened", "pleural calcification"],
    "Pneumonia":         ["pneumonia", "pneumonic", "infectious process",
                          "infection", "lobar pneumonia",
                          "bronchopneumonia"],
    "Pneumothorax":      ["pneumothorax", "tension pneumothorax",
                          "air in pleural", "collapsed lung",
                          "air leak", "visceral pleural line"],
    "Support Devices":   ["support device", "support devices", "pacemaker",
                          "catheter", "central line", "picc line",
                          "endotracheal tube", "ett", "chest tube",
                          "nasogastric tube", "ng tube", "tracheostomy",
                          "line placement", "port-a-cath",
                          "swan-ganz", "icd", "defibrillator"],
    "No Finding":        ["no finding", "no findings", "normal study",
                          "unremarkable", "no acute",
                          "no significant abnormality", "clear lungs",
                          "lungs are clear", "normal chest",
                          "within normal limits"],
}


def _is_negated(text_lower: str, match_start: int) -> bool:
    """Return True if the match at *match_start* is preceded by a negation phrase."""
    window = text_lower[max(0, match_start - _NEGATION_WINDOW): match_start]
    return any(neg in window for neg in _NEGATION_PHRASES)


def extract_pathologies_from_text(text: str) -> List[str]:
    """
    Extract pathology mentions from text using keyword matching with
    expanded synonyms and aliases.

    Args:
        text: Natural language text

    Returns:
        List of detected pathologies (canonical DenseNet-121 names)
    """
    text_lower = text.lower()
    found: List[str] = []

    for pathology, aliases in _PATHOLOGY_ALIASES.items():
        for alias in aliases:
            idx = text_lower.find(alias)
            if idx != -1:
                if pathology not in found:
                    found.append(pathology)
                break  # one alias match is enough per pathology

    return found


def extract_pathologies_with_polarity(text: str) -> List[tuple]:
    """
    Extract pathologies together with polarity (positive / negative mention).

    Returns:
        List of (pathology_name, is_positive) tuples.
    """
    text_lower = text.lower()
    results: List[tuple] = []

    for pathology, aliases in _PATHOLOGY_ALIASES.items():
        for alias in aliases:
            idx = text_lower.find(alias)
            if idx != -1:
                positive = not _is_negated(text_lower, idx)
                results.append((pathology, positive))
                break

    return results


def normalize_output(output: Any, tool_name: str, tool_type: str, **kwargs) -> List[CanonicalFinding]:
    """
    Main normalization function - routes to appropriate normalizer with unified confidence scoring.
    
    Args:
        output: Raw tool output
        tool_name: Name of the tool
        tool_type: Type of tool ("classification", "vqa", "segmentation", "grounding", "report", "generation")
        **kwargs: Additional tool-specific arguments:
            - prompt (str): Original prompt for VQA tools
            - samples (List[str]): Multiple samples for self-consistency scoring
            - mask_probabilities: Raw mask probabilities for segmentation
            - min_confidence_threshold (float): Minimum confidence for classification
        
    Returns:
        List of CanonicalFinding objects with calibrated confidence scores
    """
    if tool_type == "classification":
        min_threshold = kwargs.get("min_confidence_threshold", 0.01)
        return normalize_classification_output(output, tool_name, min_threshold)
    elif tool_type == "vqa":
        return normalize_vqa_output(
            output, 
            tool_name, 
            kwargs.get("prompt", ""),
            kwargs.get("samples", None)
        )
    elif tool_type == "segmentation":
        return normalize_segmentation_output(
            output, 
            tool_name,
            kwargs.get("mask_probabilities", None)
        )
    elif tool_type == "grounding":
        return normalize_grounding_output(output, tool_name, **kwargs)
    elif tool_type == "report":
        return normalize_report_output(output, tool_name, **kwargs)
    elif tool_type == "generation":
        return normalize_generation_output(output, tool_name, **kwargs)
    else:
        # Generic fallback - use VQA-style processing if text
        if isinstance(output, str):
            return normalize_vqa_output(output, tool_name, kwargs.get("prompt", ""))
        
        return [CanonicalFinding(
            pathology="unknown",
            region="unspecified",
            confidence=0.5,
            evidence_type=tool_type,
            source_tool=tool_name,
            raw_value=output,
            metadata=kwargs
        )]


def normalize_grounding_output(output: Any, tool_name: str, **kwargs) -> List[CanonicalFinding]:
    """
    Convert grounding tool output to canonical format using unified confidence scoring.
    
    Now reads confidence_data from tool metadata for pre-computed scores.
    
    Args:
        output: Raw grounding output (bounding boxes, predictions)
        tool_name: Name of the grounding tool
        **kwargs: Additional arguments (attention_map, region_mask, metadata, etc.)
        
    Returns:
        List of CanonicalFinding objects
    """
    findings = []
    pipeline = get_confidence_pipeline()
    
    # Handle tuple format (output_dict, metadata)
    metadata_dict = {}
    if isinstance(output, tuple):
        if len(output) > 1 and isinstance(output[1], dict):
            metadata_dict = output[1]
        output = output[0] if output and len(output) > 0 else {}
    
    # Handle dict output format
    if isinstance(output, dict):
        phrase = kwargs.get("phrase", "unknown")
        predictions = output.get("predictions", [])
        
        # Extract confidence_data from metadata if available
        confidence_data = metadata_dict.get("confidence_data", {})
        
        # Use pre-computed confidence from tool if available
        if confidence_data and "confidence_score" in confidence_data:
            raw_confidence = confidence_data["confidence_score"]
        else:
            # Fallback: base confidence on having predictions
            raw_confidence = 0.6 if predictions else 0.3
        
        # Get bounding boxes for metadata
        all_bboxes = []
        for pred in predictions:
            if isinstance(pred, dict) and "bounding_boxes" in pred:
                all_bboxes.extend(pred["bounding_boxes"].get("image_coordinates", []))
        
        # Use unified confidence scoring pipeline
        try:
            model_output = ModelOutput(
                task_type=TaskType.GROUNDING.value,
                raw_output=output,
                auxiliary={
                    "confidence_data": confidence_data,
                    "bounding_box": all_bboxes[0] if all_bboxes else None,
                    "confidence": raw_confidence,
                },
                model_name=tool_name
            )
            
            confidence_result = pipeline.process(model_output)
            calibrated_conf = confidence_result.calibrated_confidence
            uncertainty = confidence_result.uncertainty
            confidence_method = confidence_result.method
            
        except Exception as e:
            print(f"  ⚠️ Pipeline confidence scoring failed: {e}")
            calibrated_conf = calibrate_confidence(tool_name, raw_confidence)
            uncertainty = 1.0 - calibrated_conf
            confidence_method = "legacy"
        
        finding = CanonicalFinding(
            pathology=phrase,
            region="grounded_region",
            confidence=calibrated_conf,
            evidence_type="grounding",
            source_tool=tool_name,
            raw_value=output,
            metadata={
                "bounding_boxes": all_bboxes,
                "num_boxes": len(all_bboxes),
                "raw_confidence": raw_confidence,
                "uncertainty": uncertainty,
                "confidence_method": confidence_method,
                "has_predictions": len(predictions) > 0,
                "coverage_ratio": confidence_data.get("coverage_ratio", 0)
            }
        )
        findings.append(finding)
    
    return findings


def normalize_report_output(output: Any, tool_name: str, **kwargs) -> List[CanonicalFinding]:
    """
    Convert report generation output to canonical format using unified confidence scoring.
    
    Uses self-consistency across multiple generated reports for confidence estimation.
    Now reads confidence_data from tool metadata for pre-computed scores.
    
    Args:
        output: Raw report output (text, findings, impressions)
        tool_name: Name of the report generation tool
        **kwargs: Additional arguments:
            - reports (List[str]): Multiple generated reports for self-consistency
        
    Returns:
        List of CanonicalFinding objects
    """
    findings = []
    pipeline = get_confidence_pipeline()
    
    # Handle tuple format (report_text, metadata)
    metadata_dict = {}
    if isinstance(output, tuple):
        if len(output) > 1 and isinstance(output[1], dict):
            metadata_dict = output[1]
        output = output[0] if output and len(output) > 0 else ""
    
    # Extract report text
    if isinstance(output, dict):
        report_text = output.get("findings", "") + " " + output.get("impression", "")
        if not report_text.strip():
            report_text = str(output)
    else:
        report_text = str(output)
    
    # Extract confidence_data from metadata if available
    confidence_data = metadata_dict.get("confidence_data", {})
    
    # Get multiple reports for self-consistency from confidence_data or kwargs
    reports = kwargs.get("reports", None)
    if not reports and confidence_data:
        # Try to extract from confidence_data
        findings_samples = confidence_data.get("findings", {}).get("samples", [])
        if findings_samples:
            reports = findings_samples
    
    # Use unified confidence scoring pipeline
    try:
        model_output = ModelOutput(
            task_type=TaskType.REPORT.value,
            raw_output=report_text,
            auxiliary={
                "confidence_data": confidence_data,
                "reports": reports
            },
            model_name=tool_name
        )
        
        confidence_result = pipeline.process(model_output)
        calibrated_conf = confidence_result.calibrated_confidence
        uncertainty = confidence_result.uncertainty
        confidence_method = confidence_result.method
        
    except Exception as e:
        print(f"  ⚠️ Pipeline confidence scoring failed: {e}")
        # Use pre-computed overall_confidence if available
        if confidence_data and "overall_confidence" in confidence_data:
            calibrated_conf = confidence_data["overall_confidence"]
            uncertainty = 1.0 - calibrated_conf
            confidence_method = "precomputed_fallback"
        else:
            calibrated_conf = 0.5
            uncertainty = 0.5
            confidence_method = "default"
    
    # Extract pathologies from report
    pathologies = extract_pathologies_from_text(report_text)
    
    if pathologies:
        for pathology in pathologies:
            finding = CanonicalFinding(
                pathology=pathology,
                region="global",
                confidence=calibrated_conf,
                evidence_type="report",
                source_tool=tool_name,
                raw_value=report_text[:500],  # Truncate for storage
                metadata={
                    "uncertainty": uncertainty,
                    "confidence_method": confidence_method,
                    "full_report_length": len(report_text),
                    "findings_consistency": confidence_data.get("findings", {}).get("consistency_score"),
                    "impression_consistency": confidence_data.get("impression", {}).get("consistency_score")
                }
            )
            findings.append(finding)
    else:
        # Create general finding
        finding = CanonicalFinding(
            pathology="report_assessment",
            region="global",
            confidence=calibrated_conf,
            evidence_type="report",
            source_tool=tool_name,
            raw_value=report_text[:500],
            metadata={
                "uncertainty": uncertainty,
                "confidence_method": confidence_method
            }
        )
        findings.append(finding)
    
    return findings


def normalize_generation_output(output: Any, tool_name: str, **kwargs) -> List[CanonicalFinding]:
    """
    Convert image generation output to canonical format using unified confidence scoring.
    
    Now reads confidence_data from tool metadata for pre-computed consistency scores.
    
    Args:
        output: Raw generation output (image path, quality metrics)
        tool_name: Name of the generation tool
        **kwargs: Additional arguments (clip_score, classifier_agreement, etc.)
        
    Returns:
        List of CanonicalFinding objects
    """
    findings = []
    pipeline = get_confidence_pipeline()
    
    # Handle tuple format (output_dict, metadata)
    metadata_dict = {}
    if isinstance(output, tuple):
        if len(output) > 1 and isinstance(output[1], dict):
            metadata_dict = output[1]
        output = output[0] if output and len(output) > 0 else {}
    
    # Extract confidence_data from metadata if available
    confidence_data = metadata_dict.get("confidence_data", {})
    
    # Use unified confidence scoring pipeline
    try:
        model_output = ModelOutput(
            task_type=TaskType.GENERATION.value,
            raw_output=output,
            auxiliary={
                "confidence_data": confidence_data,
                "clip_score": kwargs.get("clip_score"),
                "classifier_agreement": kwargs.get("classifier_agreement"),
                "generation_quality": kwargs.get("generation_quality")
            },
            model_name=tool_name
        )
        
        confidence_result = pipeline.process(model_output)
        calibrated_conf = confidence_result.calibrated_confidence
        uncertainty = confidence_result.uncertainty
        confidence_method = confidence_result.method
        
    except Exception as e:
        print(f"  ⚠️ Pipeline confidence scoring failed: {e}")
        # Use pre-computed consistency_score if available
        if confidence_data and "consistency_score" in confidence_data:
            calibrated_conf = confidence_data["consistency_score"]
            uncertainty = 1.0 - calibrated_conf
            confidence_method = "precomputed_fallback"
        else:
            calibrated_conf = 0.5
            uncertainty = 0.5
            confidence_method = "default"
    
    prompt = kwargs.get("prompt", metadata_dict.get("prompt", "unknown_prompt"))
    
    finding = CanonicalFinding(
        pathology="generated_image",
        region="full_image",
        confidence=calibrated_conf,
        evidence_type="generation",
        source_tool=tool_name,
        raw_value=str(output)[:200],
        metadata={
            "prompt": prompt,
            "uncertainty": uncertainty,
            "confidence_method": confidence_method,
            "consistency_score": confidence_data.get("consistency_score"),
            "avg_pixel_similarity": confidence_data.get("avg_pixel_similarity"),
            "num_samples": confidence_data.get("num_samples"),
            "clip_score": kwargs.get("clip_score"),
            "classifier_agreement": kwargs.get("classifier_agreement")
        }
    )
    findings.append(finding)
    
    return findings
