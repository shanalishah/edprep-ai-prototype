#!/usr/bin/env python3
"""
Quality Essay Scorer - Production-Ready Model with Monitoring
Addresses overfitting, bias, and implements quality controls
"""

import joblib
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityEssayScorer:
    """
    Production-ready essay scorer with quality controls, monitoring, and bias detection
    """
    
    def __init__(self):
        # Fix path resolution - go up from backend/app/models to project root, then to models
        self.models_dir = Path(__file__).parent.parent.parent.parent / "models"
        self.quality_models = {}
        self.performance_history = []
        self.bias_detector = BiasDetector()
        self.quality_validator = QualityValidator()
        self.is_loaded = False
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_mae': 0.8,
            'max_overfitting_ratio': 0.1,
            'min_r2': 0.15,
            'max_bias_score': 0.2
        }
        
        self._load_quality_models()
    
    def _load_quality_models(self):
        """Load the best performing models with quality validation"""
        try:
            logger.info("🔍 Loading quality models...")
            
            # Load Wide Neural Network (best performer, no overfitting)
            nn_model_path = self.models_dir / "neural_network_wide_nn.pkl"
            logger.info(f"Looking for model at: {nn_model_path}")
            if nn_model_path.exists():
                model_data = torch.load(nn_model_path, map_location='cpu', weights_only=False)
                
                # Reconstruct the model architecture
                # Exclude 'essay' and 'overall_band_score' from feature columns
                feature_columns = [col for col in model_data['feature_columns'] if col not in ['essay', 'overall_band_score']]
                input_size = len(feature_columns)
                self.quality_models['wide_nn'] = {
                    'model': self._create_wide_nn(input_size),
                    'scaler': model_data['scaler'],
                    'feature_columns': feature_columns,
                    'performance': {
                        'mae': 1.023,
                        'r2': 0.110,
                        'overfitting_ratio': 0.0  # No overfitting detected
                    }
                }
                
                # Load the trained weights
                self.quality_models['wide_nn']['model'].load_state_dict(model_data['model_state_dict'])
                self.quality_models['wide_nn']['model'].eval()
                
                logger.info("✅ Wide Neural Network loaded successfully")
            else:
                logger.warning("⚠️ Wide Neural Network model not found")
            
            # Load fallback models for ensemble
            self._load_fallback_models()
            
            self.is_loaded = True
            logger.info("✅ Quality models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading quality models: {e}")
            self.is_loaded = False
    
    def _create_wide_nn(self, input_size):
        """Recreate the wide neural network architecture"""
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def _load_fallback_models(self):
        """Load fallback models for robustness"""
        try:
            # Load basic Random Forest as fallback (with overfitting awareness)
            rf_path = self.models_dir / "Random Forest_model.pkl"
            if rf_path.exists():
                self.quality_models['fallback_rf'] = {
                    'model': joblib.load(rf_path),
                    'performance': {
                        'mae': 0.156,
                        'r2': 0.122,
                        'overfitting_ratio': 0.696,  # High overfitting - use with caution
                        'warning': 'High overfitting detected - use only as fallback'
                    }
                }
                logger.info("⚠️ Fallback Random Forest loaded (with overfitting warning)")
        except Exception as e:
            logger.warning(f"⚠️ Could not load fallback models: {e}")
    
    def _extract_quality_features(self, essay: str, prompt: str, task_type: str) -> pd.DataFrame:
        """Extract features with quality validation"""
        try:
            # Basic features
            word_count = len(essay.split())
            sentence_count = len(re.split(r'[.!?]', essay))
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Advanced features
            complex_words = len([w for w in essay.split() if len(w) > 6])
            complexity_ratio = complex_words / word_count if word_count > 0 else 0
            
            # Cohesion features
            cohesive_markers = ['however', 'therefore', 'moreover', 'furthermore', 'consequently']
            cohesion_score = sum(essay.lower().count(marker) for marker in cohesive_markers)
            
            # Quality indicators
            quality_score = self._calculate_quality_score(essay, word_count, task_type)
            
            features = {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length,
                'complexity_ratio': complexity_ratio,
                'cohesion_score': cohesion_score,
                'quality_score': quality_score
            }
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"❌ Error extracting features: {e}")
            # Return default features
            return pd.DataFrame([{
                'word_count': 250,
                'sentence_count': 10,
                'avg_sentence_length': 25,
                'complexity_ratio': 0.3,
                'cohesion_score': 2,
                'quality_score': 0.5
            }])
    
    def _calculate_quality_score(self, essay: str, word_count: int, task_type: str) -> float:
        """Calculate essay quality score based on multiple factors"""
        try:
            quality_score = 0.5  # Base score
            
            # Word count check
            if task_type == "Task 2":
                if word_count >= 250:
                    quality_score += 0.2
                elif word_count < 150:
                    quality_score -= 0.3
            elif task_type == "Task 1":
                if word_count >= 150:
                    quality_score += 0.2
                elif word_count < 100:
                    quality_score -= 0.3
            
            # Structure check (paragraphs)
            paragraphs = essay.count('\n\n') + 1
            if paragraphs >= 3:
                quality_score += 0.1
            elif paragraphs < 2:
                quality_score -= 0.2
            
            # Coherence check (linking words)
            linking_words = ['first', 'second', 'third', 'however', 'therefore', 'moreover', 'furthermore']
            linking_count = sum(essay.lower().count(word) for word in linking_words)
            if linking_count >= 3:
                quality_score += 0.1
            elif linking_count == 0:
                quality_score -= 0.1
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating quality score: {e}")
            return 0.5
    
    def score_essay_quality(self, essay: str, prompt: str, task_type: str = "Task 2") -> Dict[str, Any]:
        """
        Score essay with quality controls and monitoring
        """
        if not self.is_loaded:
            logger.error("❌ Quality models not loaded")
            return self._get_fallback_scores(essay, task_type)
        
        try:
            # Input validation
            validation_result = self.quality_validator.validate_input(essay, prompt, task_type)
            if not validation_result['valid']:
                logger.warning(f"⚠️ Input validation failed: {validation_result['reason']}")
                return self._get_fallback_scores(essay, task_type)
            
            # Extract features
            features_df = self._extract_quality_features(essay, prompt, task_type)
            
            # Get predictions from primary model (Wide NN)
            primary_scores = self._get_primary_prediction(features_df, essay, prompt, task_type)
            
            # Get fallback prediction for comparison
            fallback_scores = self._get_fallback_prediction(features_df, essay, prompt, task_type)
            
            # Ensemble prediction with quality weighting
            final_scores = self._ensemble_predictions(primary_scores, fallback_scores, features_df)
            
            # Bias detection
            bias_result = self.bias_detector.detect_bias(essay, final_scores, task_type)
            
            # Quality assessment
            quality_assessment = self._assess_prediction_quality(final_scores, features_df, bias_result)
            
            # Log performance
            self._log_prediction(essay, final_scores, quality_assessment)
            
            return {
                'scores': final_scores,
                'quality_assessment': quality_assessment,
                'bias_detection': bias_result,
                'model_confidence': self._calculate_confidence(features_df, final_scores),
                'recommendations': self._generate_recommendations(final_scores, quality_assessment, bias_result)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in quality scoring: {e}")
            return self._get_fallback_scores(essay, task_type)
    
    def _get_primary_prediction(self, features_df: pd.DataFrame, essay: str, prompt: str, task_type: str) -> Dict[str, float]:
        """Get prediction from primary model (Wide NN)"""
        try:
            if 'wide_nn' not in self.quality_models:
                raise Exception("Primary model not available")
            
            model_data = self.quality_models['wide_nn']
            
            # Prepare features
            feature_columns = [col for col in model_data['feature_columns'] if col != 'overall_band_score']
            X = features_df[feature_columns].values
            
            # Scale features
            X_scaled = model_data['scaler'].transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            
            # Get prediction
            with torch.no_grad():
                prediction = model_data['model'](X_tensor).item()
            
            # Convert to band scores (1-9 scale)
            band_score = max(1.0, min(9.0, prediction))
            
            # Generate individual criterion scores
            scores = self._generate_criterion_scores(band_score, essay, features_df)
            
            return scores
            
        except Exception as e:
            logger.error(f"❌ Error in primary prediction: {e}")
            raise
    
    def _get_fallback_prediction(self, features_df: pd.DataFrame, essay: str, prompt: str, task_type: str) -> Dict[str, float]:
        """Get fallback prediction with overfitting awareness"""
        try:
            if 'fallback_rf' not in self.quality_models:
                # Use rule-based fallback
                return self._get_rule_based_scores(essay, task_type)
            
            model_data = self.quality_models['fallback_rf']
            
            # Apply overfitting penalty
            overfitting_penalty = model_data['performance']['overfitting_ratio'] * 0.5
            
            # Get prediction
            prediction = model_data['model'].predict(features_df)[0]
            
            # Apply penalty for overfitting
            adjusted_score = prediction - overfitting_penalty
            band_score = max(1.0, min(9.0, adjusted_score))
            
            scores = self._generate_criterion_scores(band_score, essay, features_df)
            
            return scores
            
        except Exception as e:
            logger.error(f"❌ Error in fallback prediction: {e}")
            return self._get_rule_based_scores(essay, task_type)
    
    def _ensemble_predictions(self, primary_scores: Dict[str, float], fallback_scores: Dict[str, float], features_df: pd.DataFrame) -> Dict[str, float]:
        """Ensemble predictions with quality weighting"""
        try:
            # Weight primary model more heavily (70% vs 30%)
            primary_weight = 0.7
            fallback_weight = 0.3
            
            # Adjust weights based on quality indicators
            quality_score = features_df['quality_score'].iloc[0]
            if quality_score > 0.7:
                primary_weight = 0.8  # Trust primary model more for high-quality essays
            elif quality_score < 0.3:
                fallback_weight = 0.5  # Use more fallback for low-quality essays
            
            ensemble_scores = {}
            for criterion in ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range', 'overall_band_score']:
                primary_val = primary_scores.get(criterion, 6.0)
                fallback_val = fallback_scores.get(criterion, 6.0)
                
                ensemble_val = (primary_val * primary_weight) + (fallback_val * fallback_weight)
                ensemble_scores[criterion] = round(ensemble_val * 2) / 2  # Round to nearest 0.5
            
            return ensemble_scores
            
        except Exception as e:
            logger.error(f"❌ Error in ensemble prediction: {e}")
            return primary_scores
    
    def _generate_criterion_scores(self, overall_score: float, essay: str, features_df: pd.DataFrame) -> Dict[str, float]:
        """Generate individual criterion scores based on overall score and essay features"""
        try:
            # Base scores around overall score
            base_scores = {
                'task_achievement': overall_score,
                'coherence_cohesion': overall_score,
                'lexical_resource': overall_score,
                'grammatical_range': overall_score,
                'overall_band_score': overall_score
            }
            
            # Adjust based on features
            word_count = features_df['word_count'].iloc[0]
            complexity_ratio = features_df['complexity_ratio'].iloc[0]
            cohesion_score = features_df['cohesion_score'].iloc[0]
            
            # Task Achievement adjustments
            if word_count < 200:
                base_scores['task_achievement'] -= 1.0
            elif word_count > 400:
                base_scores['task_achievement'] += 0.5
            
            # Lexical Resource adjustments
            if complexity_ratio > 0.4:
                base_scores['lexical_resource'] += 0.5
            elif complexity_ratio < 0.2:
                base_scores['lexical_resource'] -= 0.5
            
            # Coherence & Cohesion adjustments
            if cohesion_score > 3:
                base_scores['coherence_cohesion'] += 0.5
            elif cohesion_score == 0:
                base_scores['coherence_cohesion'] -= 0.5
            
            # Ensure scores are within valid range
            for criterion in base_scores:
                base_scores[criterion] = max(1.0, min(9.0, base_scores[criterion]))
                base_scores[criterion] = round(base_scores[criterion] * 2) / 2
            
            return base_scores
            
        except Exception as e:
            logger.error(f"❌ Error generating criterion scores: {e}")
            return {
                'task_achievement': overall_score,
                'coherence_cohesion': overall_score,
                'lexical_resource': overall_score,
                'grammatical_range': overall_score,
                'overall_band_score': overall_score
            }
    
    def _assess_prediction_quality(self, scores: Dict[str, float], features_df: pd.DataFrame, bias_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of the prediction"""
        try:
            quality_score = features_df['quality_score'].iloc[0]
            overall_score = scores['overall_band_score']
            
            # Quality indicators
            quality_indicators = {
                'input_quality': 'high' if quality_score > 0.7 else 'medium' if quality_score > 0.4 else 'low',
                'score_consistency': self._check_score_consistency(scores),
                'bias_risk': bias_result['risk_level'],
                'model_confidence': self._calculate_confidence(features_df, scores)
            }
            
            # Overall quality assessment
            if quality_indicators['input_quality'] == 'high' and quality_indicators['bias_risk'] == 'low':
                overall_quality = 'high'
            elif quality_indicators['input_quality'] == 'low' or quality_indicators['bias_risk'] == 'high':
                overall_quality = 'low'
            else:
                overall_quality = 'medium'
            
            return {
                'overall_quality': overall_quality,
                'indicators': quality_indicators,
                'recommendations': self._get_quality_recommendations(quality_indicators)
            }
            
        except Exception as e:
            logger.error(f"❌ Error assessing prediction quality: {e}")
            return {
                'overall_quality': 'medium',
                'indicators': {'input_quality': 'medium', 'score_consistency': 'medium', 'bias_risk': 'medium', 'model_confidence': 0.5},
                'recommendations': ['Manual review recommended']
            }
    
    def _check_score_consistency(self, scores: Dict[str, float]) -> str:
        """Check if scores are consistent across criteria"""
        try:
            criterion_scores = [scores[c] for c in ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range']]
            score_std = np.std(criterion_scores)
            
            if score_std < 0.5:
                return 'high'
            elif score_std < 1.0:
                return 'medium'
            else:
                return 'low'
        except:
            return 'medium'
    
    def _calculate_confidence(self, features_df: pd.DataFrame, scores: Dict[str, float]) -> float:
        """Calculate model confidence based on features and scores"""
        try:
            # Base confidence
            confidence = 0.7
            
            # Adjust based on quality score
            quality_score = features_df['quality_score'].iloc[0]
            confidence += (quality_score - 0.5) * 0.3
            
            # Adjust based on score consistency
            consistency = self._check_score_consistency(scores)
            if consistency == 'high':
                confidence += 0.1
            elif consistency == 'low':
                confidence -= 0.1
            
            return max(0.0, min(1.0, confidence))
        except:
            return 0.5
    
    def _generate_recommendations(self, scores: Dict[str, float], quality_assessment: Dict[str, Any], bias_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on scores and quality assessment"""
        recommendations = []
        
        # Score-based recommendations
        overall_score = scores['overall_band_score']
        if overall_score < 5.0:
            recommendations.append("Consider additional practice with IELTS writing fundamentals")
        elif overall_score > 7.5:
            recommendations.append("Excellent work! Focus on fine-tuning advanced writing skills")
        
        # Quality-based recommendations
        if quality_assessment['overall_quality'] == 'low':
            recommendations.append("Manual review recommended due to low prediction quality")
        
        # Bias-based recommendations
        if bias_result['risk_level'] == 'high':
            recommendations.append("Bias detected - consider human review")
        
        return recommendations
    
    def _get_quality_recommendations(self, indicators: Dict[str, Any]) -> List[str]:
        """Get recommendations based on quality indicators"""
        recommendations = []
        
        if indicators['input_quality'] == 'low':
            recommendations.append("Essay quality is low - consider improving structure and content")
        
        if indicators['score_consistency'] == 'low':
            recommendations.append("Scores are inconsistent - manual review recommended")
        
        if indicators['bias_risk'] == 'high':
            recommendations.append("High bias risk detected - human review recommended")
        
        return recommendations
    
    def _get_fallback_scores(self, essay: str, task_type: str) -> Dict[str, Any]:
        """Get fallback scores when models fail"""
        scores = self._get_rule_based_scores(essay, task_type)
        return {
            'scores': scores,
            'quality_assessment': {'overall_quality': 'low', 'indicators': {}, 'recommendations': ['Fallback scoring used']},
            'bias_detection': {'risk_level': 'medium', 'details': 'Fallback mode'},
            'model_confidence': 0.3,
            'recommendations': ['Manual review recommended - model unavailable']
        }
    
    def _get_rule_based_scores(self, essay: str, task_type: str) -> Dict[str, float]:
        """Rule-based scoring as final fallback"""
        word_count = len(essay.split())
        
        # Basic scoring based on word count and structure
        if word_count < 150:
            base_score = 4.0
        elif word_count < 250:
            base_score = 5.5
        elif word_count < 350:
            base_score = 6.5
        else:
            base_score = 7.0
        
        return {
            'task_achievement': base_score,
            'coherence_cohesion': base_score,
            'lexical_resource': base_score,
            'grammatical_range': base_score,
            'overall_band_score': base_score
        }
    
    def _log_prediction(self, essay: str, scores: Dict[str, float], quality_assessment: Dict[str, Any]):
        """Log prediction for monitoring and improvement"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'essay_length': len(essay.split()),
                'scores': scores,
                'quality_assessment': quality_assessment,
                'model_version': 'quality_v1.0'
            }
            
            self.performance_history.append(log_entry)
            
            # Keep only last 1000 entries
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            logger.error(f"❌ Error logging prediction: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        try:
            if not self.performance_history:
                return {'message': 'No performance data available'}
            
            # Calculate metrics
            recent_predictions = self.performance_history[-100:]  # Last 100 predictions
            
            avg_confidence = np.mean([p.get('quality_assessment', {}).get('indicators', {}).get('model_confidence', 0.5) for p in recent_predictions])
            avg_quality = np.mean([1 if p.get('quality_assessment', {}).get('overall_quality') == 'high' else 0.5 if p.get('quality_assessment', {}).get('overall_quality') == 'medium' else 0 for p in recent_predictions])
            
            return {
                'total_predictions': len(self.performance_history),
                'recent_avg_confidence': round(avg_confidence, 3),
                'recent_avg_quality': round(avg_quality, 3),
                'model_status': 'operational' if self.is_loaded else 'degraded',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance metrics: {e}")
            return {'error': str(e)}


class BiasDetector:
    """Detect bias in predictions"""
    
    def detect_bias(self, essay: str, scores: Dict[str, float], task_type: str) -> Dict[str, Any]:
        """Detect potential bias in the prediction"""
        try:
            bias_indicators = []
            risk_level = 'low'
            
            # Check for length bias
            word_count = len(essay.split())
            if word_count < 100 and scores['overall_band_score'] > 6.0:
                bias_indicators.append('Length bias: Short essay scored too high')
                risk_level = 'medium'
            elif word_count > 500 and scores['overall_band_score'] < 5.0:
                bias_indicators.append('Length bias: Long essay scored too low')
                risk_level = 'medium'
            
            # Check for score distribution bias
            if scores['overall_band_score'] > 8.0:
                # Check if all criteria are similarly high (potential over-scoring)
                criterion_scores = [scores[c] for c in ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range']]
                if all(score > 7.5 for score in criterion_scores):
                    bias_indicators.append('Potential over-scoring bias')
                    risk_level = 'high'
            
            return {
                'risk_level': risk_level,
                'indicators': bias_indicators,
                'details': f'Detected {len(bias_indicators)} bias indicators'
            }
            
        except Exception as e:
            logger.error(f"❌ Error detecting bias: {e}")
            return {'risk_level': 'medium', 'indicators': ['Bias detection failed'], 'details': str(e)}


class QualityValidator:
    """Validate input quality"""
    
    def validate_input(self, essay: str, prompt: str, task_type: str) -> Dict[str, Any]:
        """Validate input for quality scoring"""
        try:
            issues = []
            
            # Check essay length
            word_count = len(essay.split())
            if word_count < 50:
                issues.append('Essay too short (less than 50 words)')
            elif word_count > 1000:
                issues.append('Essay too long (more than 1000 words)')
            
            # Check prompt
            if len(prompt.strip()) < 10:
                issues.append('Prompt too short')
            
            # Check for obvious issues
            if essay.strip() == '':
                issues.append('Empty essay')
            
            if len(essay.split('.')) < 2:
                issues.append('Essay appears to be a single sentence')
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'reason': '; '.join(issues) if issues else 'Valid input'
            }
            
        except Exception as e:
            logger.error(f"❌ Error validating input: {e}")
            return {'valid': False, 'issues': ['Validation failed'], 'reason': str(e)}
