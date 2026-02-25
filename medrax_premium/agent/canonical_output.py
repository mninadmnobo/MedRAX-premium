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
    
    print("  🔍 DEBUG: normalize_classification_output called")
    print(f"     Input type: {type(output)}")
    print(f"     Input value (first 200 chars): {str(output)[:200]}")
    
    # Extract the actual output dict (handle tuple format)
    if isinstance(output, tuple):
        print(f"     Detected tuple with {len(output)} elements")
        output_dict = output[0] if output and len(output) > 0 else {}
        metadata = output[1] if len(output) > 1 else {}
    elif isinstance(output, dict):
        print(f"     Detected dict with {len(output)} keys")
        output_dict = output
        metadata = {}
    else:
        # Can't parse this format
        print(f"  ⚠️ Warning: Unexpected classification output format: {type(output)}")
        return findings
    
    # Validate output_dict is actually a dict
    if not isinstance(output_dict, dict):
        print(f"  ⚠️ Warning: Expected dict, got {type(output_dict)}")
        return findings
    
    print(f"     Output dict has {len(output_dict)} entries")
    
    filtered_count = 0
    created_count = 0
    
    for pathology, prob in output_dict.items():
        # Skip non-numeric entries (accept both Python and numpy numeric types)
        try:
            prob_float = float(prob)  # Convert to Python float (works for numpy types too)
        except (TypeError, ValueError):
            print(f"     ⏭️ Skipping {pathology}: non-numeric value {type(prob)}")
            continue
        
        # Only include findings with meaningful probability
        if prob_float < min_confidence_threshold:
            filtered_count += 1
            continue
        
        print(f"     ✅ Processing {pathology}: {prob_float:.4f}")
        
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
            
            print(f"        Raw: {prob_float:.4f} -> Calibrated: {calibrated_conf:.4f}")
            print(f"        Uncertainty: {uncertainty:.4f} | Method: {confidence_result.method}")
            
        except Exception as e:
            print(f"        ❌ Pipeline calibration failed: {e}, using legacy calibration")
            calibrated_conf = calibrate_confidence(tool_name, prob_float)
            uncertainty = 1.0 - calibrated_conf
        
        try:
            finding = CanonicalFinding(
                pathology=pathology,
                region="global",  # Classification is global, not localized
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
            created_count += 1
            print(f"        ✅ Created finding #{created_count}")
        except Exception as e:
            print(f"        ❌ Failed to create CanonicalFinding: {e}")
    
    print(f"     Filtered out {filtered_count} findings below {min_confidence_threshold*100:.0f}% threshold")
    print(f"     ✅ Created {len(findings)} canonical findings")
    
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
    
    Args:
        output: Raw text output from VQA model (can be dict, tuple, or string)
        tool_name: Name of the VQA tool
        prompt: Original prompt/question
        samples: Optional multiple samples for self-consistency scoring
        
    Returns:
        List of CanonicalFinding objects with calibrated confidence scores
    """
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
    
    # Handle dict with error
    if isinstance(output, dict) and "error" in output:
        return []
    
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
        print(f"  ⚠️ Pipeline confidence scoring failed: {e}, using legacy method")
        raw_confidence = estimate_text_confidence(text)
        calibrated_conf = calibrate_confidence(tool_name, raw_confidence)
        uncertainty = 1.0 - calibrated_conf
        confidence_method = "keyword_analysis_legacy"
    
    # Simple pathology extraction (can be improved with NER)
    pathologies = extract_pathologies_from_text(text)
    
    findings = []
    if pathologies:
        for pathology in pathologies:
            finding = CanonicalFinding(
                pathology=pathology,
                region="unspecified",  # Region extraction can be improved
                confidence=calibrated_conf,
                evidence_type="vqa",
                source_tool=tool_name,
                raw_value=text,
                metadata={
                    "prompt": prompt,
                    "full_response": text,
                    "confidence_estimation_method": confidence_method,
                    "uncertainty": uncertainty,
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
                "full_response": text,
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
                print(f"  ⚠️ Pipeline confidence scoring failed for {region}: {e}")
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


def extract_pathologies_from_text(text: str) -> List[str]:
    """
    Extract pathology mentions from text using keyword matching.
    
    TODO: Replace with proper Named Entity Recognition (NER) model.
    
    Args:
        text: Natural language text
        
    Returns:
        List of detected pathologies
    """
    # Standard pathology list
    PATHOLOGIES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
        "Emphysema", "Fibrosis", "Fracture", "Hernia", "Infiltration",
        "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax",
        "Support Devices", "No Finding"
    ]
    
    text_lower = text.lower()
    found = []
    
    for pathology in PATHOLOGIES:
        if pathology.lower() in text_lower:
            found.append(pathology)
    
    return found


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
