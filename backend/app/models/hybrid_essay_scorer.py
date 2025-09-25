"""
Hybrid Essay Scorer: Combines Official Rubric with ML Models
Dynamically adjusts weights based on essay characteristics and performance data
"""

import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional
import json

from .essay_scorer import EssayScorer
from .ml_essay_scorer import MLEssayScorer
from .production_essay_scorer import ProductionEssayScorer

class HybridEssayScorer:
    """
    Hybrid scoring system that combines official IELTS rubric with ML models
    Uses dynamic weight adjustment based on essay characteristics and performance data
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.rubric_scorer = EssayScorer()
        self.ml_scorer = MLEssayScorer()
        self.production_scorer = ProductionEssayScorer()
        
        # Default configuration
        self.config = {
            "default_rubric_weight": 0.6,  # Default: 60% rubric, 40% ML
            "min_rubric_weight": 0.2,      # Minimum rubric weight
            "max_rubric_weight": 0.9,      # Maximum rubric weight
            "quality_thresholds": {
                "high": {"min_words": 300, "min_sentences": 15},
                "medium": {"min_words": 200, "min_sentences": 10},
                "low": {"min_words": 100, "min_sentences": 5}
            },
            "weight_adjustments": {
                "high_quality": 0.1,      # Increase rubric weight for high quality
                "low_quality": -0.1,      # Decrease rubric weight for low quality
                "short_essay": -0.15,     # Decrease rubric weight for short essays
                "long_essay": 0.1,        # Increase rubric weight for long essays
                "complex_prompt": 0.05,   # Increase rubric weight for complex prompts
            }
        }
        
        # Load custom configuration if provided
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        
        # Performance tracking
        self.performance_history = []
        
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                self.config.update(custom_config)
                print(f"✅ Loaded custom configuration from {config_path}")
        except Exception as e:
            print(f"⚠️ Error loading config: {e}. Using default configuration.")
    
    def save_config(self, config_path: str):
        """Save current configuration to JSON file"""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
                print(f"✅ Configuration saved to {config_path}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def _analyze_essay_characteristics(self, essay: str, prompt: str) -> Dict[str, Any]:
        """
        Analyze essay characteristics to determine optimal scoring approach
        """
        word_count = len(essay.split())
        sentence_count = len(re.split(r'[.!?]', essay))
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Calculate complexity metrics
        complex_words = len([word for word in essay.split() if len(word) > 6])
        complexity_ratio = complex_words / word_count if word_count > 0 else 0
        
        # Prompt complexity (simple heuristic)
        prompt_complexity = len(prompt.split()) / 20  # Normalize by typical prompt length
        
        # Determine quality level
        quality_level = "medium"
        if word_count >= self.config["quality_thresholds"]["high"]["min_words"] and \
           sentence_count >= self.config["quality_thresholds"]["high"]["min_sentences"]:
            quality_level = "high"
        elif word_count < self.config["quality_thresholds"]["low"]["min_words"] or \
             sentence_count < self.config["quality_thresholds"]["low"]["min_sentences"]:
            quality_level = "low"
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "complexity_ratio": complexity_ratio,
            "prompt_complexity": prompt_complexity,
            "quality_level": quality_level
        }
    
    def _calculate_dynamic_weight(self, characteristics: Dict[str, Any]) -> float:
        """
        Calculate dynamic weight based on essay characteristics
        """
        base_weight = self.config["default_rubric_weight"]
        adjustment = 0.0
        
        # Quality-based adjustments
        if characteristics["quality_level"] == "high":
            adjustment += self.config["weight_adjustments"]["high_quality"]
        elif characteristics["quality_level"] == "low":
            adjustment += self.config["weight_adjustments"]["low_quality"]
        
        # Length-based adjustments
        if characteristics["word_count"] < 200:
            adjustment += self.config["weight_adjustments"]["short_essay"]
        elif characteristics["word_count"] > 400:
            adjustment += self.config["weight_adjustments"]["long_essay"]
        
        # Complexity-based adjustments
        if characteristics["prompt_complexity"] > 1.0:
            adjustment += self.config["weight_adjustments"]["complex_prompt"]
        
        # Apply adjustment and ensure within bounds
        final_weight = base_weight + adjustment
        final_weight = max(self.config["min_rubric_weight"], 
                          min(self.config["max_rubric_weight"], final_weight))
        
        return final_weight
    
    def _get_available_scorers(self) -> Dict[str, Any]:
        """
        Get information about available scoring methods
        """
        return {
            "rubric": True,  # Always available
            "ml_basic": self.ml_scorer.is_loaded,
            "ml_production": self.production_scorer.is_loaded
        }
    
    def score_essay_hybrid(self, essay: str, prompt: str, task_type: str = "Task 2", 
                          force_rubric_weight: Optional[float] = None) -> Dict[str, Any]:
        """
        Score essay using hybrid approach with dynamic weight adjustment
        """
        # Analyze essay characteristics
        characteristics = self._analyze_essay_characteristics(essay, prompt)
        
        # Determine rubric weight
        if force_rubric_weight is not None:
            rubric_weight = force_rubric_weight
        else:
            rubric_weight = self._calculate_dynamic_weight(characteristics)
        
        ml_weight = 1.0 - rubric_weight
        
        # Get rubric-based scores
        rubric_scores = self.rubric_scorer.score_essay(prompt, essay, task_type)
        
        # Get ML-based scores (prefer production over basic)
        ml_scores = {}
        ml_method = "none"
        
        if self.production_scorer.is_loaded:
            try:
                ml_scores = self.production_scorer.score_essay_production(essay, prompt, task_type)
                ml_method = "production"
            except Exception as e:
                print(f"⚠️ Production ML scoring failed: {e}")
        
        if not ml_scores and self.ml_scorer.is_loaded:
            try:
                ml_scores = self.ml_scorer.score_essay_ml(essay, prompt, task_type)
                ml_method = "basic"
            except Exception as e:
                print(f"⚠️ Basic ML scoring failed: {e}")
        
        # Calculate hybrid scores
        hybrid_scores = {}
        if ml_scores:
            for criterion in ["task_achievement", "coherence_cohesion", "lexical_resource", "grammatical_range", "overall_band_score"]:
                if criterion in rubric_scores and criterion in ml_scores:
                    hybrid_scores[criterion] = (
                        rubric_weight * rubric_scores[criterion] + 
                        ml_weight * ml_scores[criterion]
                    )
        else:
            # Fallback to rubric-only if ML fails
            hybrid_scores = rubric_scores.copy()
            rubric_weight = 1.0
            ml_weight = 0.0
        
        # Round to nearest 0.5 band score
        for criterion in hybrid_scores:
            hybrid_scores[criterion] = round(hybrid_scores[criterion] * 2) / 2
        
        # Prepare detailed response
        result = {
            "scores": hybrid_scores,
            "scoring_details": {
                "rubric_weight": rubric_weight,
                "ml_weight": ml_weight,
                "ml_method": ml_method,
                "characteristics": characteristics,
                "available_scorers": self._get_available_scorers()
            },
            "individual_scores": {
                "rubric": rubric_scores,
                "ml": ml_scores if ml_scores else None
            }
        }
        
        # Track performance for learning
        self._track_performance(result, characteristics)
        
        return result
    
    def _track_performance(self, result: Dict[str, Any], characteristics: Dict[str, Any]):
        """
        Track scoring performance for future optimization
        """
        performance_record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "characteristics": characteristics,
            "scoring_details": result["scoring_details"],
            "scores": result["scores"]
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only last 1000 records to prevent memory issues
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of scoring performance
        """
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        df = pd.DataFrame(self.performance_history)
        
        # Calculate statistics
        summary = {
            "total_essays_scored": len(df),
            "average_rubric_weight": df["scoring_details"].apply(lambda x: x["rubric_weight"]).mean(),
            "weight_distribution": df["scoring_details"].apply(lambda x: x["rubric_weight"]).value_counts().to_dict(),
            "ml_method_usage": df["scoring_details"].apply(lambda x: x["ml_method"]).value_counts().to_dict(),
            "quality_level_distribution": df["characteristics"].apply(lambda x: x["quality_level"]).value_counts().to_dict()
        }
        
        return summary
    
    def optimize_weights(self, target_accuracy: float = 0.8) -> Dict[str, Any]:
        """
        Optimize weights based on performance history
        This is a placeholder for future ML-based optimization
        """
        if len(self.performance_history) < 50:
            return {"message": "Insufficient data for optimization. Need at least 50 scored essays."}
        
        # Simple optimization based on quality level performance
        df = pd.DataFrame(self.performance_history)
        
        # Group by quality level and calculate average weights
        quality_weights = df.groupby(df["characteristics"].apply(lambda x: x["quality_level"]))[
            "scoring_details"
        ].apply(lambda x: x.apply(lambda y: y["rubric_weight"]).mean()).to_dict()
        
        # Update configuration with optimized weights
        optimized_config = self.config.copy()
        for quality, weight in quality_weights.items():
            if quality == "high":
                optimized_config["weight_adjustments"]["high_quality"] = weight - self.config["default_rubric_weight"]
            elif quality == "low":
                optimized_config["weight_adjustments"]["low_quality"] = weight - self.config["default_rubric_weight"]
        
        return {
            "optimized_weights": quality_weights,
            "recommended_config": optimized_config,
            "current_config": self.config
        }
    
    def export_performance_data(self, filepath: str):
        """
        Export performance history to JSON file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
                print(f"✅ Performance data exported to {filepath}")
        except Exception as e:
            print(f"❌ Error exporting performance data: {e}")
