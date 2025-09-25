"""
Comprehensive Model Training Pipeline for IELTS Essay Scoring
This pipeline trains and compares multiple ML models to find the best performer
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Text processing
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Try to download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class IELTSModelTrainer:
    def __init__(self, data_dir: str, models_dir: str):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.scalers = {}
        self.vectorizers = {}
        self.models = {}
        self.results = {}
        
        # Load datasets
        self.train_df = pd.read_csv(self.data_dir / "train_dataset.csv")
        self.val_df = pd.read_csv(self.data_dir / "val_dataset.csv")
        self.test_df = pd.read_csv(self.data_dir / "test_dataset.csv")
        
        print(f"📊 Loaded datasets:")
        print(f"   Train: {len(self.train_df)} essays")
        print(f"   Validation: {len(self.val_df)} essays")
        print(f"   Test: {len(self.test_df)} essays")
    
    def extract_text_features(self, essays: List[str]) -> pd.DataFrame:
        """Extract comprehensive text features from essays"""
        print("🔧 Extracting text features...")
        
        features = []
        
        for essay in essays:
            if not essay or pd.isna(essay):
                features.append(self._get_empty_features())
                continue
            
            essay = str(essay)
            
            # Basic features
            word_count = len(essay.split())
            char_count = len(essay)
            sentence_count = len(sent_tokenize(essay))
            paragraph_count = len([p for p in essay.split('\n\n') if p.strip()])
            
            # Word-level features
            words = word_tokenize(essay.lower())
            unique_words = len(set(words))
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Sentence-level features
            sentences = sent_tokenize(essay)
            avg_sentence_length = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
            
            # Vocabulary features
            vocabulary_richness = unique_words / word_count if word_count > 0 else 0
            
            # Grammar and style features
            complex_sentences = len(re.findall(r'\b(because|although|while|whereas|if|when|since|as)\b', essay.lower()))
            linking_words = len(re.findall(r'\b(however|therefore|moreover|furthermore|additionally|on the other hand|in contrast|similarly|likewise|firstly|secondly|finally|in addition|as a result)\b', essay.lower()))
            
            # Academic vocabulary
            academic_words = len(re.findall(r'\b(significant|substantial|considerable|essential|crucial|fundamental|comprehensive|extensive|effective|efficient|consequently|subsequently|furthermore|moreover|nevertheless)\b', essay.lower()))
            
            # Punctuation features
            comma_count = essay.count(',')
            period_count = essay.count('.')
            question_count = essay.count('?')
            exclamation_count = essay.count('!')
            
            # Capitalization features
            capital_ratio = sum(1 for c in essay if c.isupper()) / len(essay) if essay else 0
            
            # Repetition features
            word_freq = Counter(words)
            max_word_freq = max(word_freq.values()) if word_freq else 0
            repetition_ratio = max_word_freq / word_count if word_count > 0 else 0
            
            features.append({
                'word_count': word_count,
                'char_count': char_count,
                'sentence_count': sentence_count,
                'paragraph_count': paragraph_count,
                'unique_words': unique_words,
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'vocabulary_richness': vocabulary_richness,
                'complex_sentences': complex_sentences,
                'linking_words': linking_words,
                'academic_words': academic_words,
                'comma_count': comma_count,
                'period_count': period_count,
                'question_count': question_count,
                'exclamation_count': exclamation_count,
                'capital_ratio': capital_ratio,
                'repetition_ratio': repetition_ratio
            })
        
        return pd.DataFrame(features)
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return empty features for null essays"""
        return {
            'word_count': 0, 'char_count': 0, 'sentence_count': 0, 'paragraph_count': 0,
            'unique_words': 0, 'avg_word_length': 0, 'avg_sentence_length': 0,
            'vocabulary_richness': 0, 'complex_sentences': 0, 'linking_words': 0,
            'academic_words': 0, 'comma_count': 0, 'period_count': 0,
            'question_count': 0, 'exclamation_count': 0, 'capital_ratio': 0,
            'repetition_ratio': 0
        }
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training"""
        print("🔧 Preparing features...")
        
        # Extract text features
        text_features = self.extract_text_features(df['essay'].tolist())
        
        # Combine with existing features
        feature_columns = [
            'word_count', 'essay_length', 'avg_word_length', 'sentence_count',
            'paragraph_count', 'complex_sentences', 'linking_words'
        ]
        
        existing_features = df[feature_columns].values if all(col in df.columns for col in feature_columns) else np.array([]).reshape(len(df), 0)
        
        # Combine all features
        if existing_features.size > 0:
            X = np.hstack([text_features.values, existing_features])
        else:
            X = text_features.values
        
        # Target variables (scores)
        target_columns = ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range', 'overall_band_score']
        y = df[target_columns].values
        
        return X, y
    
    def train_model(self, model_name: str, model, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train a single model and return results"""
        print(f"🤖 Training {model_name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Calculate metrics for each target
            target_names = ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range', 'overall_band_score']
            
            results = {
                'model_name': model_name,
                'train_metrics': {},
                'val_metrics': {},
                'overall_train_metrics': {},
                'overall_val_metrics': {}
            }
            
            # Calculate metrics for each target
            for i, target in enumerate(target_names):
                train_mse = mean_squared_error(y_train[:, i], y_train_pred[:, i])
                train_mae = mean_absolute_error(y_train[:, i], y_train_pred[:, i])
                train_r2 = r2_score(y_train[:, i], y_train_pred[:, i])
                
                val_mse = mean_squared_error(y_val[:, i], y_val_pred[:, i])
                val_mae = mean_absolute_error(y_val[:, i], y_val_pred[:, i])
                val_r2 = r2_score(y_val[:, i], y_val_pred[:, i])
                
                results['train_metrics'][target] = {
                    'mse': train_mse,
                    'mae': train_mae,
                    'r2': train_r2
                }
                
                results['val_metrics'][target] = {
                    'mse': val_mse,
                    'mae': val_mae,
                    'r2': val_r2
                }
            
            # Calculate overall metrics (average across all targets)
            train_mse_overall = np.mean([results['train_metrics'][target]['mse'] for target in target_names])
            train_mae_overall = np.mean([results['train_metrics'][target]['mae'] for target in target_names])
            train_r2_overall = np.mean([results['train_metrics'][target]['r2'] for target in target_names])
            
            val_mse_overall = np.mean([results['val_metrics'][target]['mse'] for target in target_names])
            val_mae_overall = np.mean([results['val_metrics'][target]['mae'] for target in target_names])
            val_r2_overall = np.mean([results['val_metrics'][target]['r2'] for target in target_names])
            
            results['overall_train_metrics'] = {
                'mse': train_mse_overall,
                'mae': train_mae_overall,
                'r2': train_r2_overall
            }
            
            results['overall_val_metrics'] = {
                'mse': val_mse_overall,
                'mae': val_mae_overall,
                'r2': val_r2_overall
            }
            
            # Save model
            model_path = self.models_dir / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"   ✅ {model_name} - Val R²: {val_r2_overall:.4f}, Val MAE: {val_mae_overall:.4f}")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error training {model_name}: {e}")
            return None
    
    def train_all_models(self):
        """Train multiple models and compare performance"""
        print("🚀 Training Multiple Models for IELTS Essay Scoring")
        print("=" * 60)
        
        # Prepare features
        X_train, y_train = self.prepare_features(self.train_df)
        X_val, y_val = self.prepare_features(self.val_df)
        
        print(f"📊 Feature matrix shape: {X_train.shape}")
        print(f"📊 Target matrix shape: {y_train.shape}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Save scaler
        with open(self.models_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        # Define models to train
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # Train each model
        for model_name, model in models.items():
            result = self.train_model(model_name, model, X_train_scaled, y_train, X_val_scaled, y_val)
            if result:
                self.results[model_name] = result
                self.models[model_name] = model
        
        # Find best model
        best_model_name = min(self.results.keys(), 
                            key=lambda x: self.results[x]['overall_val_metrics']['mse'])
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Validation MSE: {self.results[best_model_name]['overall_val_metrics']['mse']:.4f}")
        print(f"   Validation MAE: {self.results[best_model_name]['overall_val_metrics']['mae']:.4f}")
        print(f"   Validation R²: {self.results[best_model_name]['overall_val_metrics']['r2']:.4f}")
        
        return best_model_name
    
    def evaluate_on_test_set(self, best_model_name: str):
        """Evaluate the best model on test set"""
        print(f"\n🧪 Evaluating {best_model_name} on Test Set")
        print("=" * 60)
        
        # Load scaler
        with open(self.models_dir / "scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        # Load best model
        model_path = self.models_dir / f"{best_model_name}_model.pkl"
        with open(model_path, 'rb') as f:
            best_model = pickle.load(f)
        
        # Prepare test features
        X_test, y_test = self.prepare_features(self.test_df)
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_test_pred = best_model.predict(X_test_scaled)
        
        # Calculate test metrics
        target_names = ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range', 'overall_band_score']
        
        test_results = {}
        for i, target in enumerate(target_names):
            mse = mean_squared_error(y_test[:, i], y_test_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
            r2 = r2_score(y_test[:, i], y_test_pred[:, i])
            
            test_results[target] = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"📊 {target}:")
            print(f"   MSE: {mse:.4f}")
            print(f"   MAE: {mae:.4f}")
            print(f"   R²: {r2:.4f}")
        
        # Overall test metrics
        overall_mse = np.mean([test_results[target]['mse'] for target in target_names])
        overall_mae = np.mean([test_results[target]['mae'] for target in target_names])
        overall_r2 = np.mean([test_results[target]['r2'] for target in target_names])
        
        print(f"\n🎯 Overall Test Performance:")
        print(f"   MSE: {overall_mse:.4f}")
        print(f"   MAE: {overall_mae:.4f}")
        print(f"   R²: {overall_r2:.4f}")
        
        # Save test results
        test_results['overall'] = {
            'mse': overall_mse,
            'mae': overall_mae,
            'r2': overall_r2
        }
        
        with open(self.models_dir / "test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)
        
        return test_results
    
    def save_training_summary(self):
        """Save comprehensive training summary"""
        print("\n💾 Saving Training Summary...")
        
        summary = {
            'dataset_info': {
                'train_size': len(self.train_df),
                'val_size': len(self.val_df),
                'test_size': len(self.test_df),
                'total_features': len(self.train_df.columns)
            },
            'model_results': self.results,
            'best_model': min(self.results.keys(), 
                            key=lambda x: self.results[x]['overall_val_metrics']['mse']),
            'training_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(self.models_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Training summary saved to {self.models_dir / 'training_summary.json'}")

def main():
    """Main training function"""
    data_dir = "/Users/shan/Desktop/Work/Projects/EdPrep AI/edprep-ai-prototype/data"
    models_dir = "/Users/shan/Desktop/Work/Projects/EdPrep AI/edprep-ai-prototype/models"
    
    trainer = IELTSModelTrainer(data_dir, models_dir)
    
    # Train all models
    best_model_name = trainer.train_all_models()
    
    # Evaluate on test set
    test_results = trainer.evaluate_on_test_set(best_model_name)
    
    # Save summary
    trainer.save_training_summary()
    
    print("\n🎉 Model Training Completed Successfully!")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📁 Models saved to: {models_dir}")
    
    return trainer, best_model_name, test_results

if __name__ == "__main__":
    main()
