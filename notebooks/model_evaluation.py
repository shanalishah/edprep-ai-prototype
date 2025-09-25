#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for EdPrep AI
Checks for overfitting, bias, variance, and other common ML issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))

import pandas as pd
import numpy as np
import joblib
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, learning_curve
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation for detecting overfitting and other issues
    """
    
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent / "models"
        self.data_path = Path(__file__).parent.parent / "data" / "full_dataset.csv"
        self.results = {}
        
    def load_data(self):
        """Load and prepare data for evaluation"""
        print("📊 Loading dataset for evaluation...")
        
        try:
            df = pd.read_csv(self.data_path, encoding='utf-8')
        except:
            df = pd.read_csv(self.data_path, encoding='latin-1')
        
        # Clean data
        df = df.dropna(subset=['essay', 'overall_band_score'])
        df = df[df['overall_band_score'].between(1, 9)]
        df = df[df['essay'].str.len() > 50]
        df = df[df['essay'].str.len() < 10000]
        
        print(f"✅ Loaded {len(df)} essays for evaluation")
        return df
    
    def extract_features(self, df):
        """Extract features for traditional ML models"""
        features = []
        
        for idx, row in df.iterrows():
            essay = row['essay']
            
            # Basic features
            word_count = len(essay.split())
            sentence_count = len(essay.split('.'))
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Advanced features
            complex_words = len([w for w in essay.split() if len(w) > 6])
            complexity_ratio = complex_words / word_count if word_count > 0 else 0
            
            # Cohesion features
            cohesive_markers = ['however', 'therefore', 'moreover', 'furthermore', 'consequently']
            cohesion_score = sum(essay.lower().count(marker) for marker in cohesive_markers)
            
            features.append({
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length,
                'complexity_ratio': complexity_ratio,
                'cohesion_score': cohesion_score,
                'overall_band_score': row['overall_band_score']
            })
        
        return pd.DataFrame(features)
    
    def evaluate_traditional_models(self, df):
        """Evaluate traditional ML models for overfitting"""
        print("\n🔍 Evaluating Traditional ML Models...")
        
        # Extract features
        features_df = self.extract_features(df)
        X = features_df.drop(['overall_band_score'], axis=1)
        y = features_df['overall_band_score']
        
        # Load models (skip production models that have multi-output issues)
        models_to_evaluate = [
            'Random Forest_model.pkl'
        ]
        
        traditional_results = {}
        
        for model_file in models_to_evaluate:
            model_path = self.models_dir / model_file
            if not model_path.exists():
                print(f"⚠️ Model not found: {model_file}")
                continue
                
            print(f"\n📊 Evaluating {model_file}...")
            
            try:
                model = joblib.load(model_path)
                
                # Cross-validation for overfitting detection
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Learning curve analysis
                train_sizes, train_scores, val_scores = learning_curve(
                    model, X, y, cv=3, n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, 5),
                    scoring='neg_mean_absolute_error'
                )
                
                train_mae = -train_scores.mean(axis=1)
                val_mae = -val_scores.mean(axis=1)
                
                # Calculate overfitting metrics
                final_train_mae = train_mae[-1]
                final_val_mae = val_mae[-1]
                overfitting_gap = final_val_mae - final_train_mae
                overfitting_ratio = overfitting_gap / final_train_mae if final_train_mae > 0 else 0
                
                # Bias-Variance Analysis
                bias = final_train_mae
                variance = cv_std
                
                traditional_results[model_file] = {
                    'cv_mae': cv_mae,
                    'cv_std': cv_std,
                    'final_train_mae': final_train_mae,
                    'final_val_mae': final_val_mae,
                    'overfitting_gap': overfitting_gap,
                    'overfitting_ratio': overfitting_ratio,
                    'bias': bias,
                    'variance': variance,
                    'learning_curve': {
                        'train_sizes': train_sizes.tolist(),
                        'train_mae': train_mae.tolist(),
                        'val_mae': val_mae.tolist()
                    }
                }
                
                # Overfitting assessment
                if overfitting_ratio > 0.2:
                    status = "🔴 HIGH OVERFITTING"
                elif overfitting_ratio > 0.1:
                    status = "🟡 MODERATE OVERFITTING"
                else:
                    status = "🟢 LOW OVERFITTING"
                
                print(f"  {status}")
                print(f"  CV MAE: {cv_mae:.3f} ± {cv_std:.3f}")
                print(f"  Train MAE: {final_train_mae:.3f}")
                print(f"  Val MAE: {final_val_mae:.3f}")
                print(f"  Overfitting Gap: {overfitting_gap:.3f}")
                print(f"  Overfitting Ratio: {overfitting_ratio:.3f}")
                
            except Exception as e:
                print(f"❌ Error evaluating {model_file}: {e}")
                continue
        
        return traditional_results
    
    def evaluate_neural_networks(self, df):
        """Evaluate neural network models for overfitting"""
        print("\n🧠 Evaluating Neural Network Models...")
        
        # Extract features
        features_df = self.extract_features(df)
        X = features_df.drop(['overall_band_score'], axis=1).values
        y = features_df['overall_band_score'].values
        
        # Load neural network models
        nn_models = [
            'neural_network_simple_nn.pkl',
            'neural_network_deep_nn.pkl',
            'neural_network_wide_nn.pkl'
        ]
        
        nn_results = {}
        
        for model_file in nn_models:
            model_path = self.models_dir / model_file
            if not model_path.exists():
                print(f"⚠️ Model not found: {model_file}")
                continue
                
            print(f"\n📊 Evaluating {model_file}...")
            
            try:
                # Load model data with weights_only=False for compatibility
                model_data = torch.load(model_path, map_location='cpu', weights_only=False)
                
                # Extract training history if available
                if 'train_losses' in model_data and 'val_losses' in model_data:
                    train_losses = model_data['train_losses']
                    val_losses = model_data['val_losses']
                    
                    # Calculate overfitting metrics
                    final_train_loss = train_losses[-1]
                    final_val_loss = val_losses[-1]
                    overfitting_gap = final_val_loss - final_train_loss
                    overfitting_ratio = overfitting_gap / final_train_loss if final_train_loss > 0 else 0
                    
                    # Find minimum validation loss (best epoch)
                    min_val_loss = min(val_losses)
                    min_val_epoch = val_losses.index(min_val_loss)
                    best_train_loss = train_losses[min_val_epoch]
                    
                    # Early stopping analysis
                    early_stopping_gap = final_val_loss - min_val_loss
                    
                    nn_results[model_file] = {
                        'final_train_loss': final_train_loss,
                        'final_val_loss': final_val_loss,
                        'min_val_loss': min_val_loss,
                        'best_epoch': min_val_epoch,
                        'overfitting_gap': overfitting_gap,
                        'overfitting_ratio': overfitting_ratio,
                        'early_stopping_gap': early_stopping_gap,
                        'train_losses': train_losses,
                        'val_losses': val_losses
                    }
                    
                    # Overfitting assessment
                    if overfitting_ratio > 0.2:
                        status = "🔴 HIGH OVERFITTING"
                    elif overfitting_ratio > 0.1:
                        status = "🟡 MODERATE OVERFITTING"
                    else:
                        status = "🟢 LOW OVERFITTING"
                    
                    print(f"  {status}")
                    print(f"  Final Train Loss: {final_train_loss:.3f}")
                    print(f"  Final Val Loss: {final_val_loss:.3f}")
                    print(f"  Min Val Loss: {min_val_loss:.3f} (Epoch {min_val_epoch})")
                    print(f"  Overfitting Gap: {overfitting_gap:.3f}")
                    print(f"  Overfitting Ratio: {overfitting_ratio:.3f}")
                    print(f"  Early Stopping Gap: {early_stopping_gap:.3f}")
                    
                    # Check for training issues
                    if early_stopping_gap > 0.1:
                        print(f"  ⚠️ WARNING: Model continued training after best epoch")
                    
                    if overfitting_ratio > 0.3:
                        print(f"  ⚠️ WARNING: Severe overfitting detected")
                    
                else:
                    print(f"  ⚠️ No training history found in {model_file}")
                    
            except Exception as e:
                print(f"❌ Error evaluating {model_file}: {e}")
                continue
        
        return nn_results
    
    def analyze_data_distribution(self, df):
        """Analyze data distribution for bias detection"""
        print("\n📈 Analyzing Data Distribution...")
        
        # Score distribution
        score_dist = df['overall_band_score'].value_counts().sort_index()
        
        # Word count distribution
        word_counts = df['essay'].str.split().str.len()
        
        # Calculate statistics
        score_stats = {
            'mean': df['overall_band_score'].mean(),
            'std': df['overall_band_score'].std(),
            'min': df['overall_band_score'].min(),
            'max': df['overall_band_score'].max(),
            'median': df['overall_band_score'].median()
        }
        
        word_stats = {
            'mean': word_counts.mean(),
            'std': word_counts.std(),
            'min': word_counts.min(),
            'max': word_counts.max(),
            'median': word_counts.median()
        }
        
        # Check for class imbalance
        score_imbalance = score_dist.max() / score_dist.min()
        
        print(f"  Score Distribution:")
        print(f"    Mean: {score_stats['mean']:.2f}")
        print(f"    Std: {score_stats['std']:.2f}")
        print(f"    Range: {score_stats['min']:.1f} - {score_stats['max']:.1f}")
        print(f"    Imbalance Ratio: {score_imbalance:.2f}")
        
        print(f"  Word Count Distribution:")
        print(f"    Mean: {word_stats['mean']:.0f}")
        print(f"    Std: {word_stats['std']:.0f}")
        print(f"    Range: {word_stats['min']:.0f} - {word_stats['max']:.0f}")
        
        # Bias warnings
        if score_imbalance > 3:
            print(f"  ⚠️ WARNING: High class imbalance detected")
        
        if score_stats['std'] < 1.0:
            print(f"  ⚠️ WARNING: Low score variance may indicate bias")
        
        return {
            'score_stats': score_stats,
            'word_stats': word_stats,
            'score_imbalance': score_imbalance,
            'score_distribution': {str(k): int(v) for k, v in score_dist.to_dict().items()}
        }
    
    def generate_recommendations(self, traditional_results, nn_results, data_analysis):
        """Generate recommendations based on evaluation results"""
        print("\n💡 Recommendations:")
        
        recommendations = []
        
        # Traditional model recommendations
        for model_name, results in traditional_results.items():
            if results['overfitting_ratio'] > 0.2:
                recommendations.append(f"🔴 {model_name}: High overfitting - consider regularization or more data")
            elif results['overfitting_ratio'] > 0.1:
                recommendations.append(f"🟡 {model_name}: Moderate overfitting - consider early stopping")
            
            if results['variance'] > 0.5:
                recommendations.append(f"🟡 {model_name}: High variance - consider ensemble methods")
        
        # Neural network recommendations
        for model_name, results in nn_results.items():
            if results['overfitting_ratio'] > 0.2:
                recommendations.append(f"🔴 {model_name}: High overfitting - increase dropout or reduce model complexity")
            elif results['overfitting_ratio'] > 0.1:
                recommendations.append(f"🟡 {model_name}: Moderate overfitting - implement early stopping")
            
            if results['early_stopping_gap'] > 0.1:
                recommendations.append(f"🟡 {model_name}: Implement early stopping to prevent overtraining")
        
        # Data recommendations
        if data_analysis['score_imbalance'] > 3:
            recommendations.append("🟡 Data: High class imbalance - consider stratified sampling or data augmentation")
        
        if data_analysis['score_stats']['std'] < 1.0:
            recommendations.append("🟡 Data: Low score variance - consider data collection from diverse sources")
        
        # Print recommendations
        if recommendations:
            for rec in recommendations:
                print(f"  {rec}")
        else:
            print("  🟢 No major issues detected - models look good!")
        
        return recommendations
    
    def save_evaluation_report(self, traditional_results, nn_results, data_analysis, recommendations):
        """Save comprehensive evaluation report"""
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        report = {
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'traditional_models': convert_numpy(traditional_results),
            'neural_networks': convert_numpy(nn_results),
            'data_analysis': convert_numpy(data_analysis),
            'recommendations': recommendations,
            'summary': {
                'total_models_evaluated': len(traditional_results) + len(nn_results),
                'models_with_overfitting': sum(1 for r in traditional_results.values() if r['overfitting_ratio'] > 0.1) + 
                                         sum(1 for r in nn_results.values() if r['overfitting_ratio'] > 0.1),
                'data_issues': len([r for r in recommendations if 'Data:' in r])
            }
        }
        
        report_path = self.models_dir / "model_evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Evaluation report saved to: {report_path}")
        return report
    
    def run_comprehensive_evaluation(self):
        """Run complete model evaluation"""
        print("🔍 Starting Comprehensive Model Evaluation")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Analyze data distribution
        data_analysis = self.analyze_data_distribution(df)
        
        # Evaluate traditional models
        traditional_results = self.evaluate_traditional_models(df)
        
        # Evaluate neural networks
        nn_results = self.evaluate_neural_networks(df)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(traditional_results, nn_results, data_analysis)
        
        # Save report
        report = self.save_evaluation_report(traditional_results, nn_results, data_analysis, recommendations)
        
        print("\n✅ Comprehensive Model Evaluation Complete!")
        print("=" * 60)
        
        return report

def main():
    """Main function to run model evaluation"""
    evaluator = ModelEvaluator()
    report = evaluator.run_comprehensive_evaluation()
    
    print(f"\n📊 Evaluation Summary:")
    print(f"  Models Evaluated: {report['summary']['total_models_evaluated']}")
    print(f"  Models with Overfitting: {report['summary']['models_with_overfitting']}")
    print(f"  Data Issues: {report['summary']['data_issues']}")

if __name__ == "__main__":
    main()
