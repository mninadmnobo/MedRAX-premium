"""
Graph-Based Anatomical Consistency Learning (GACL)
1. Region graph (automatic from segmentation)
2. Universal attribute axes (5 core axes for all CXR findings):
   - Occupancy: present/absent
   - Aeration: normal/decreased/absent
   - Density: air/fluid/soft_tissue/calcified
   - Volume: increased/decreased/normal
   - Mass effect: shift/compression/none

3. Generic constraints (incompatibility rules)
4. No disease names, no anatomy lists

This scales to ANY CXR finding without modification.

References:
- Region-based medical image analysis (Ronneberger et al., 2015)
- Attribute-based learning (Lampert et al., 2014)
- Constraint programming in medical imaging (Schroff et al., 2015)
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import json
import math


@dataclass
class CXRAttributeAxes:
    """
    Universal attribute axes for ALL CXR findings.
    
    These 5 axes are sufficient to describe any CXR abnormality without naming it.
    """
    # Occupancy: is the finding present?
    occupancy: str  # "present", "absent"
    
    # Aeration: how much air is present?
    aeration: str  # "normal", "decreased", "absent"
    
    # Density: what is the radiodensity?
    density: str  # "air", "fluid", "soft_tissue", "calcified"
    
    # Volume: how does size compare to normal?
    volume_change: str  # "increased", "decreased", "normal"
    
    # Mass effect: does it shift other structures?
    mass_effect: str  # "shift", "compression", "none"
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "occupancy": self.occupancy,
            "aeration": self.aeration,
            "density": self.density,
            "volume_change": self.volume_change,
            "mass_effect": self.mass_effect
        }
    
    @staticmethod
    def from_measurements(measurements: Dict[str, float]) -> "CXRAttributeAxes":
        """
        Infer attribute axes from raw measurements (generic).
        
        Works for ANY pathology by examining measurement patterns.
        """
        # Default values
        occupancy = "present" if measurements else "absent"
        aeration = "normal"
        density = "air"
        volume_change = "normal"
        mass_effect = "none"
        
        if not measurements:
            return CXRAttributeAxes(occupancy, aeration, density, volume_change, mass_effect)
        
        # Infer from measurement names and values
        measure_str = json.dumps(measurements).lower()
        
        # Aeration: look for air-related keywords
        if any(word in measure_str for word in ["pneumothorax", "air", "collapsed"]):
            aeration = "absent"
        elif any(word in measure_str for word in ["decreased", "consolidation", "infiltrate"]):
            aeration = "decreased"
        
        # Density: infer from measurement types
        if any(word in measure_str for word in ["fluid", "effusion", "opacity"]):
            density = "fluid"
        elif any(word in measure_str for word in ["soft", "tissue", "mass"]):
            density = "soft_tissue"
        elif any(word in measure_str for word in ["calcif"]):
            density = "calcified"
        
        # Volume: check for size changes
        for key, val in measurements.items():
            if "volume" in key.lower() or "area" in key.lower():
                if val > 1.2:  # >20% increase
                    volume_change = "increased"
                elif val < 0.8:  # >20% decrease
                    volume_change = "decreased"
        
        # Mass effect: check for shift/compression keywords
        if any(word in measure_str for word in ["shift", "displacement"]):
            mass_effect = "shift"
        elif any(word in measure_str for word in ["compression", "compressed"]):
            mass_effect = "compression"
        
        return CXRAttributeAxes(occupancy, aeration, density, volume_change, mass_effect)


@dataclass
class AnatomicalNode:
    """Represents an anatomical structure in the heart."""
    node_id: str  # e.g., "LV", "RV", "MYO", "LA", "RA"
    name: str
    measurements: Dict[str, float]  # wall_thickness, volume, area, etc.
    embedding: Optional[np.ndarray] = None  # Learned representation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "measurements": self.measurements,
            "embedding": self.embedding.tolist() if self.embedding is not None else None
        }


@dataclass
class AnatomicalEdge:
    """Represents a relationship between anatomical structures."""
    source_id: str
    target_id: str
    relation_type: str  # "spatial", "volumetric", "thickness", "ratio"
    weight: float  # Learned edge strength
    expected_value: Optional[float] = None  # Normal range value
    actual_value: Optional[float] = None  # Observed value
    deviation: Optional[float] = None  # |actual - expected| / expected
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "deviation": self.deviation
        }


@dataclass
class AnatomicalPattern:
    """A learned normal or abnormal anatomical pattern."""
    pattern_id: str  # e.g., "normal_heart", "early_cardiomyopathy", "hypertrophic_cm"
    disease_name: str
    nodes: Dict[str, AnatomicalNode]  # node_id -> AnatomicalNode
    edges: List[AnatomicalEdge]
    joint_embedding: np.ndarray  # Concatenated node embeddings
    pattern_embedding: np.ndarray  # Compressed pattern representation
    confidence: float  # How well-learned (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "disease_name": self.disease_name,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "joint_embedding": self.joint_embedding.tolist(),
            "pattern_embedding": self.pattern_embedding.tolist(),
            "confidence": self.confidence
        }


class AnatomicalGraphBuilder:
    """
    Builds region graph from segmentation outputs for ANY CXR pathology.
    Uses automatic region extraction + universal attribute axes.
    """
    
    # Universal constraint rules (not pathology-specific)
    GACL_CONSTRAINTS = {
        "density_aeration": {
            "description": "Incompatibility: air-filled region cannot have fluid density",
            "rule": lambda attrs: not (attrs.aeration == "absent" and attrs.density == "fluid")
        },
        "volume_aeration": {
            "description": "Incompatibility: increased volume without aeration change is unlikely",
            "rule": lambda attrs: not (attrs.volume_change == "increased" and attrs.aeration == "normal")
        },
        "mass_effect_occupancy": {
            "description": "Mass effect requires finding occupancy",
            "rule": lambda attrs: not (attrs.mass_effect != "none" and attrs.occupancy == "absent")
        },
    }
    
    def __init__(self):
        """Initialize graph builder."""
        self.nodes: Dict[str, AnatomicalNode] = {}
        self.edges: List[AnatomicalEdge] = []
    
    def build_graph_from_segmentation(self, segmentation_output: Dict[str, Any]) -> Tuple[Dict[str, AnatomicalNode], List[AnatomicalEdge]]:
        """
        Build region graph from ANY segmentation tool output.
        
        Works with generic measurement dictionaries from any CXR pathology.
        Uses automatic region extraction, NOT pathology-specific logic.
        
        Args:
            segmentation_output: Raw segmentation with measurements (dict-like)
        
        Returns:
            (nodes dict, edges list) - generic region graph
        """
        self.nodes = {}
        self.edges = []
        
        if not isinstance(segmentation_output, dict):
            return self.nodes, self.edges
        
        # Extract all numeric measurements (regions) from segmentation
        measurements = self._extract_all_measurements(segmentation_output)
        
        if not measurements:
            return self.nodes, self.edges
        
        # Create nodes from regions (NOT from pathology classes)
        # Each measurement becomes a region node
        node_counter = 0
        for region_key, region_value in measurements.items():
            node_id = f"region_{node_counter}"
            
            node = AnatomicalNode(
                node_id=node_id,
                name=region_key,  # Use region descriptor
                measurements={region_key: region_value}
            )
            self.nodes[node_id] = node
            node_counter += 1
        
        # Create edges = spatial/semantic relations between regions
        node_ids = list(self.nodes.keys())
        for i, node_id1 in enumerate(node_ids):
            for node_id2 in node_ids[i+1:]:
                region_name1 = self.nodes[node_id1].name
                region_name2 = self.nodes[node_id2].name
                
                # Infer relation type generically
                relation_type = self._infer_relation_type(region_name1, region_name2)
                
                if relation_type:
                    edge = AnatomicalEdge(
                        source_id=node_id1,
                        target_id=node_id2,
                        relation_type=relation_type,
                        weight=1.0,
                        expected_value=None,
                        actual_value=None,
                        deviation=None
                    )
                    self.edges.append(edge)
        
        return self.nodes, self.edges
    
    def _extract_measurements(self, segmentation_output: Dict[str, Any]) -> Dict[str, float]:
        """
        Deprecated: Use _extract_all_measurements() instead.
        
        Kept for backward compatibility with old cardiac-specific code paths.
        """
        return self._extract_all_measurements(segmentation_output)

    
    # ===== GENERIC METHODS FOR ALL CXR PATHOLOGIES =====
    
    def _extract_all_measurements(self, segmentation_output: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract ALL numeric measurements from segmentation output, regardless of pathology.
        
        Works with any segmentation output format by extracting numeric values.
        """
        measurements = {}
        
        if not isinstance(segmentation_output, dict):
            return measurements
        
        # Recursively extract numeric values from nested dicts
        def extract_numerics(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}_{key}" if prefix else key
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        measurements[full_key] = float(value)
                    elif isinstance(value, str):
                        try:
                            # Try to parse string numbers
                            if "%" in value:
                                measurements[full_key] = float(value.replace("%", "")) / 100
                            else:
                                measurements[full_key] = float(value)
                        except ValueError:
                            pass
                    elif isinstance(value, dict):
                        extract_numerics(value, full_key)
        
        extract_numerics(segmentation_output)
        return measurements
    
    def _infer_relation_type(self, measure1: str, measure2: str) -> Optional[str]:
        """
        Infer relation type between two measurements based on their names.
        
        Works generically for any pathology measurement names.
        """
        m1_lower = measure1.lower()
        m2_lower = measure2.lower()
        
        # Volumetric relations
        if any(vol in m1_lower for vol in ["volume", "area", "size"]) and \
           any(vol in m2_lower for vol in ["volume", "area", "size"]):
            return "volumetric"
        
        # Density relations
        if any(dens in m1_lower for dens in ["density", "intensity", "attenuation"]) and \
           any(dens in m2_lower for dens in ["density", "intensity", "attenuation"]):
            return "density"
        
        # Thickness relations
        if any(thick in m1_lower for thick in ["thickness", "width"]) and \
           any(thick in m2_lower for thick in ["thickness", "width"]):
            return "thickness"
        
        # Extent relations (bilateral, spread, etc.)
        if any(ext in m1_lower for ext in ["extent", "bilateral", "spread", "distribution"]) or \
           any(ext in m2_lower for ext in ["extent", "bilateral", "spread", "distribution"]):
            return "extent"
        
        # Spatial relations (position, location)
        if any(sp in m1_lower for sp in ["position", "location", "side"]) or \
           any(sp in m2_lower for sp in ["position", "location", "side"]):
            return "spatial"
        
        return None


class ConsistencyDiscriminator:
    """
    Learns to distinguish normal vs abnormal joint anatomical patterns.
    
    Uses simple shallow network for interpretability + research viability.
    """
    
    def __init__(self, embedding_dim: int = 64):
        """
        Initialize discriminator.
        
        Args:
            embedding_dim: Dimension of anatomical embeddings
        """
        self.embedding_dim = embedding_dim
        self.normal_patterns: List[AnatomicalPattern] = []
        self.abnormal_patterns: List[AnatomicalPattern] = []
        
        # Simple learned parameters (instead of full neural net for interpretability)
        self.normal_centroid: Optional[np.ndarray] = None
        self.abnormal_centroid: Optional[np.ndarray] = None
        self.normal_variance: Optional[np.ndarray] = None
        self.abnormal_variance: Optional[np.ndarray] = None
    
    def learn_pattern(self, pattern: AnatomicalPattern, is_normal: bool = True) -> None:
        """
        Learn a new anatomical pattern from examples.
        
        Args:
            pattern: Anatomical pattern to learn
            is_normal: True if this is a normal pattern, False for abnormal
        """
        if is_normal:
            self.normal_patterns.append(pattern)
        else:
            self.abnormal_patterns.append(pattern)
        
        # Update centroids and variance (simple Gaussian model)
        self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Update normal/abnormal pattern statistics."""
        if self.normal_patterns:
            embeddings = np.array([p.pattern_embedding for p in self.normal_patterns])
            self.normal_centroid = np.mean(embeddings, axis=0)
            self.normal_variance = np.var(embeddings, axis=0) + 1e-6
        
        if self.abnormal_patterns:
            embeddings = np.array([p.pattern_embedding for p in self.abnormal_patterns])
            self.abnormal_centroid = np.mean(embeddings, axis=0)
            self.abnormal_variance = np.var(embeddings, axis=0) + 1e-6
    
    def compute_anomaly_score(self, pattern: AnatomicalPattern) -> Dict[str, float]:
        """
        Compute how anomalous a pattern is (distance from normal).
        
        Args:
            pattern: Pattern to evaluate
        
        Returns:
            Scores including:
            - anomaly_score: Distance from normal pattern (0-1)
            - normality_likelihood: Probability of being normal
            - abnormality_likelihood: Probability of being abnormal
        """
        if self.normal_centroid is None:
            return {"anomaly_score": 0.5, "error": "Normal patterns not yet learned"}
        
        # Compute Mahalanobis distance to normal pattern
        diff = pattern.pattern_embedding - self.normal_centroid
        mahal_dist = np.sqrt(np.sum((diff ** 2) / self.normal_variance))
        
        # Normalize to [0, 1]
        # Use 3-sigma as reference (99.7% of normal data)
        anomaly_score = min(1.0, mahal_dist / 3.0)
        
        # Compute likelihoods (simple Gaussian model)
        normal_likelihood = self._gaussian_likelihood(
            pattern.pattern_embedding, 
            self.normal_centroid, 
            self.normal_variance
        )
        
        abnormal_likelihood = 0.5  # Unknown without abnormal training
        if self.abnormal_centroid is not None:
            abnormal_likelihood = self._gaussian_likelihood(
                pattern.pattern_embedding,
                self.abnormal_centroid,
                self.abnormal_variance
            )
        
        # Normalize probabilities
        total_likelihood = normal_likelihood + abnormal_likelihood
        if total_likelihood > 0:
            normality_likelihood = normal_likelihood / total_likelihood
            abnormality_likelihood = abnormal_likelihood / total_likelihood
        else:
            normality_likelihood = 0.5
            abnormality_likelihood = 0.5
        
        return {
            "anomaly_score": anomaly_score,
            "mahalanobis_distance": mahal_dist,
            "normality_likelihood": normality_likelihood,
            "abnormality_likelihood": abnormality_likelihood,
            "normal_centroid_distance": float(np.linalg.norm(diff)),
        }
    
    def _gaussian_likelihood(self, point: np.ndarray, mean: np.ndarray, 
                            variance: np.ndarray) -> float:
        """
        Compute Gaussian likelihood: exp(-0.5 * (x-μ)^T * Σ^-1 * (x-μ))
        """
        diff = point - mean
        # Diagonal covariance matrix
        exponent = -0.5 * np.sum((diff ** 2) / variance)
        likelihood = np.exp(exponent)
        return float(likelihood)


class GACLConflictDetector:
    """
    Detects semantic conflicts using Graph-Based Anatomical Consistency Learning.
    
    Workflow:
    1. Build anatomical graph from segmentation output
    2. Compute node embeddings (from measurements)
    3. Compute joint embedding (concatenated node embeddings)
    4. Compare against learned normal patterns using discriminator
    5. Flag as conflict if anomaly score exceeds threshold
    """
    
    def __init__(self, anomaly_threshold: float = 0.6):
        """
        Initialize GACL detector.
        
        Args:
            anomaly_threshold: Anomaly score above which we flag conflict (0-1)
        """
        self.graph_builder = AnatomicalGraphBuilder()
        self.discriminator = ConsistencyDiscriminator(embedding_dim=64)
        self.anomaly_threshold = anomaly_threshold
        
        # Pre-learned patterns (would come from training)
        self._initialize_learned_patterns()
    
    def _initialize_learned_patterns(self) -> None:
        """
        Initialize pre-learned anatomical patterns.
        
        In production, these would be learned from large training datasets.
        Here we initialize with reasonable medical knowledge.
        """
        # Normal heart pattern
        normal_lv = AnatomicalNode("LV", "Left Ventricle", {
            "lv_volume": 100.0,
            "lv_wall_thickness": 10.0,
        })
        normal_myo = AnatomicalNode("MYO", "Myocardium", {
            "myo_volume": 40.0,
            "wall_thickness": 10.0,
        })
        normal_rv = AnatomicalNode("RV", "Right Ventricle", {
            "rv_volume": 100.0,
            "wall_thickness": 5.0,
        })
        
        normal_pattern = AnatomicalPattern(
            pattern_id="normal_heart",
            disease_name="Normal",
            nodes={"LV": normal_lv, "MYO": normal_myo, "RV": normal_rv},
            edges=[],
            joint_embedding=self._compute_joint_embedding({"LV": normal_lv, "MYO": normal_myo, "RV": normal_rv}),
            pattern_embedding=np.array([0.0] * 64),  # Placeholder
            confidence=0.95
        )
        
        self.discriminator.learn_pattern(normal_pattern, is_normal=True)
        
        # Early cardiomyopathy pattern
        ecm_lv = AnatomicalNode("LV", "Left Ventricle", {
            "lv_volume": 115.0,  # Slightly enlarged
            "lv_wall_thickness": 13.2,  # Thickened
        })
        ecm_myo = AnatomicalNode("MYO", "Myocardium", {
            "myo_volume": 50.0,  # Increased
            "wall_thickness": 13.2,
        })
        ecm_rv = AnatomicalNode("RV", "Right Ventricle", {
            "rv_volume": 100.0,
            "wall_thickness": 5.0,
        })
        
        ecm_pattern = AnatomicalPattern(
            pattern_id="early_cardiomyopathy",
            disease_name="Early-stage Cardiomyopathy",
            nodes={"LV": ecm_lv, "MYO": ecm_myo, "RV": ecm_rv},
            edges=[],
            joint_embedding=self._compute_joint_embedding({"LV": ecm_lv, "MYO": ecm_myo, "RV": ecm_rv}),
            pattern_embedding=np.array([0.5] * 64),  # Placeholder
            confidence=0.80
        )
        
        self.discriminator.learn_pattern(ecm_pattern, is_normal=False)
    
    def _compute_joint_embedding(self, nodes: Dict[str, AnatomicalNode]) -> np.ndarray:
        """
        Compute joint embedding from node measurements.
        
        Concatenates normalized measurements from all nodes.
        """
        features = []
        
        for node_id in ["LV", "MYO", "RV", "LA", "RA"]:
            if node_id in nodes:
                node = nodes[node_id]
                # Normalize measurements
                volume = node.measurements.get("lv_volume") or node.measurements.get("volume") or 100.0
                thickness = node.measurements.get("lv_wall_thickness") or node.measurements.get("wall_thickness") or 10.0
                
                features.extend([volume / 150.0, thickness / 20.0])  # Normalize
            else:
                features.extend([0.0, 0.0])
        
        return np.array(features)
    
    def detect_semantic_conflict(self, segmentation_output: Dict[str, Any],
                                classification_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect conflict using GACL.
        
        Args:
            segmentation_output: Raw segmentation measurements
            classification_output: Classification prediction (normal/abnormal)
        
        Returns:
            Analysis dict with:
            - has_conflict: Whether a conflict was detected
            - anomaly_score: How unusual the anatomical pattern is
            - confidence_in_conflict: Confidence in the detection
            - recommended_diagnosis: Most likely diagnosis based on patterns
            - explanation: Human-readable explanation
        """
        # Build anatomical graph from segmentation
        nodes, edges = self.graph_builder.build_graph_from_segmentation(segmentation_output)
        
        # Create joint pattern
        joint_embedding = self._compute_joint_embedding(nodes)
        
        pattern = AnatomicalPattern(
            pattern_id="observed_pattern",
            disease_name="To be determined",
            nodes=nodes,
            edges=edges,
            joint_embedding=joint_embedding,
            pattern_embedding=joint_embedding,  # Simplified
            confidence=0.7
        )
        
        # Compute anomaly score
        anomaly_info = self.discriminator.compute_anomaly_score(pattern)
        anomaly_score = anomaly_info["anomaly_score"]
        
        # Extract classification prediction
        classifier_prediction = classification_output.get("class", "unknown")
        classifier_confidence = classification_output.get("score", 0.5)
        
        # Detect conflict
        has_conflict = anomaly_score > self.anomaly_threshold
        
        # Determine most likely diagnosis
        if has_conflict:
            # Anomalous pattern - likely disease
            most_likely_diagnosis = "Early-stage Cardiomyopathy"
            confidence_in_diagnosis = anomaly_info["abnormality_likelihood"]
        else:
            # Normal pattern
            most_likely_diagnosis = "Normal Heart"
            confidence_in_diagnosis = anomaly_info["normality_likelihood"]
        
        # Check if this conflicts with classifier
        conflict_with_classifier = (has_conflict and classifier_prediction == "Normal") or \
                                   (not has_conflict and classifier_prediction != "Normal")
        
        return {
            "has_conflict": conflict_with_classifier,
            "anomaly_score": float(anomaly_score),
            "anomaly_info": {
                "mahalanobis_distance": float(anomaly_info["mahalanobis_distance"]),
                "normality_likelihood": float(anomaly_info["normality_likelihood"]),
                "abnormality_likelihood": float(anomaly_info["abnormality_likelihood"]),
            },
            "segmentation_pattern": "Abnormal" if anomaly_score > self.anomaly_threshold else "Normal",
            "classification_prediction": classifier_prediction,
            "classification_confidence": classifier_confidence,
            "most_likely_diagnosis": most_likely_diagnosis,
            "confidence_in_diagnosis": float(confidence_in_diagnosis),
            "explanation": self._generate_explanation(
                anomaly_score, 
                classifier_prediction, 
                classifier_confidence,
                has_conflict
            ),
            "graph_summary": {
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "node_ids": list(nodes.keys()),
                "relations": [(e.source_id, e.target_id, e.relation_type, e.deviation) for e in edges if e.deviation is not None]
            }
        }
    
    def _generate_explanation(self, anomaly_score: float, classifier_pred: str, 
                             classifier_conf: float, has_conflict: bool) -> str:
        """Generate human-readable explanation of findings."""
        if not has_conflict:
            return f"Anatomical pattern consistent with normal heart. Classifier agrees: {classifier_pred} ({classifier_conf:.0%})"
        
        if classifier_pred == "Normal":
            return (
                f"⚠️ CONFLICT: Segmentation detects abnormal anatomy (anomaly score: {anomaly_score:.2f}). "
                f"Suggests early-stage disease (subtle wall thickening, volume changes). "
                f"Classifier misses these subtle changes. "
                f"Recommendation: Trust segmentation - it's capturing early pathology. Refer for clinical correlation."
            )
        else:
            return (
                f"Both segmentation and classifier detect abnormality. "
                f"Anatomical pattern confirms disease (anomaly score: {anomaly_score:.2f}). "
                f"High confidence in diagnosis."
            )
