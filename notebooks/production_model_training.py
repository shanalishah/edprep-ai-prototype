"""
Production-Ready Model Training Pipeline for IELTS Essay Scoring
This pipeline focuses on practical improvements that can be deployed immediately
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor

# Text processing
import re
from collections import Counter
import nltk
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

class ProductionIELTSModelTrainer:
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
    
    def extract_production_features(self, essays: List[str]) -> pd.DataFrame:
        """Extract production-ready text features"""
        print("🔧 Extracting production features...")
        
        features = []
        
        for i, essay in enumerate(essays):
            if i % 5000 == 0:
                print(f"   Processing essay {i+1}/{len(essays)}")
            
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
            sentence_length_std = np.std([len(sent.split()) for sent in sentences]) if sentences else 0
            
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
            semicolon_count = essay.count(';')
            colon_count = essay.count(':')
            
            # Capitalization features
            capital_ratio = sum(1 for c in essay if c.isupper()) / len(essay) if essay else 0
            
            # Repetition features
            word_freq = Counter(words)
            max_word_freq = max(word_freq.values()) if word_freq else 0
            repetition_ratio = max_word_freq / word_count if word_count > 0 else 0
            
            # Advanced repetition analysis
            unique_word_ratio = len(set(words)) / len(words) if words else 0
            hapax_legomena = sum(1 for word, count in word_freq.items() if count == 1)
            hapax_ratio = hapax_legomena / len(words) if words else 0
            
            # Readability features (simplified)
            avg_syllables_per_word = np.mean([self._count_syllables(word) for word in words]) if words else 0
            
            # Cohesion features
            pronoun_count = len(re.findall(r'\b(i|you|he|she|it|we|they|me|him|her|us|them|my|your|his|her|its|our|their)\b', essay.lower()))
            pronoun_ratio = pronoun_count / word_count if word_count > 0 else 0
            
            # Argument structure features
            argument_indicators = len(re.findall(r'\b(firstly|secondly|thirdly|moreover|furthermore|additionally|in addition|besides|also|too)\b', essay.lower()))
            conclusion_indicators = len(re.findall(r'\b(in conclusion|to conclude|to sum up|overall|finally|ultimately)\b', essay.lower()))
            
            # Task-specific features
            opinion_indicators = len(re.findall(r'\b(i believe|i think|in my opinion|i agree|i disagree|i argue|i maintain)\b', essay.lower()))
            example_indicators = len(re.findall(r'\b(for example|for instance|such as|like|including|namely)\b', essay.lower()))
            
            features.append({
                'word_count': word_count,
                'char_count': char_count,
                'sentence_count': sentence_count,
                'paragraph_count': paragraph_count,
                'unique_words': unique_words,
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length,
                'sentence_length_std': sentence_length_std,
                'vocabulary_richness': vocabulary_richness,
                'complex_sentences': complex_sentences,
                'linking_words': linking_words,
                'academic_words': academic_words,
                'comma_count': comma_count,
                'period_count': period_count,
                'question_count': question_count,
                'exclamation_count': exclamation_count,
                'semicolon_count': semicolon_count,
                'colon_count': colon_count,
                'capital_ratio': capital_ratio,
                'repetition_ratio': repetition_ratio,
                'unique_word_ratio': unique_word_ratio,
                'hapax_ratio': hapax_ratio,
                'avg_syllables_per_word': avg_syllables_per_word,
                'pronoun_ratio': pronoun_ratio,
                'argument_indicators': argument_indicators,
                'conclusion_indicators': conclusion_indicators,
                'opinion_indicators': opinion_indicators,
                'example_indicators': example_indicators
            })
        
        return pd.DataFrame(features)
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return empty features for null essays"""
        return {
            'word_count': 0, 'char_count': 0, 'sentence_count': 0, 'paragraph_count': 0,
            'unique_words': 0, 'avg_word_length': 0, 'avg_sentence_length': 0,
            'sentence_length_std': 0, 'vocabulary_richness': 0, 'complex_sentences': 0,
            'linking_words': 0, 'academic_words': 0, 'comma_count': 0, 'period_count': 0,
            'question_count': 0, 'exclamation_count': 0, 'semicolon_count': 0,
            'colon_count': 0, 'capital_ratio': 0, 'repetition_ratio': 0,
            'unique_word_ratio': 0, 'hapax_ratio': 0, 'avg_syllables_per_word': 0,
            'pronoun_ratio': 0, 'argument_indicators': 0, 'conclusion_indicators': 0,
            'opinion_indicators': 0, 'example_indicators': 0
        }
    
    def create_tfidf_features(self, essays: List[str], max_features: int = 500) -> np.ndarray:
        """Create TF-IDF features from essays"""
        print("🔧 Creating TF-IDF features...")
        
        # Clean essays for TF-IDF
        cleaned_essays = []
        for essay in essays:
            if not essay or pd.isna(essay):
                cleaned_essays.append("")
                continue
            
            # Basic cleaning
            essay = str(essay).lower()
            essay = re.sub(r'[^\w\s]', ' ', essay)
            essay = re.sub(r'\s+', ' ', essay)
            cleaned_essays.append(essay)
        
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9
        )
        
        tfidf_features = tfidf.fit_transform(cleaned_essays)
        
        # Save vectorizer
        with open(self.models_dir / "production_tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(tfidf, f)
        
        return tfidf_features.toarray()
    
    def prepare_production_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare production features for training"""
        print("🔧 Preparing production features...")
        
        # Extract production text features
        text_features = self.extract_production_features(df['essay'].tolist())
        
        # Create TF-IDF features
        tfidf_features = self.create_tfidf_features(df['essay'].tolist())
        
        # Combine with existing features
        feature_columns = [
            'word_count', 'essay_length', 'avg_word_length', 'sentence_count',
            'paragraph_count', 'complex_sentences', 'linking_words'
        ]
        
        existing_features = df[feature_columns].values if all(col in df.columns for col in feature_columns) else np.array([]).reshape(len(df), 0)
        
        # Combine all features
        if existing_features.size > 0:
            X = np.hstack([text_features.values, existing_features, tfidf_features])
        else:
            X = np.hstack([text_features.values, tfidf_features])
        
        # Target variables (scores)
        target_columns = ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range', 'overall_band_score']
        y = df[target_columns].values
        
        print(f"📊 Production feature matrix shape: {X.shape}")
        print(f"📊 Target matrix shape: {y.shape}")
        
        return X, y
    
    def train_production_models(self):
        """Train production-ready models with proper validation"""
        print("🚀 Training Production Models for IELTS Essay Scoring")
        print("=" * 60)
        
        # Prepare features
        X_train, y_train = self.prepare_production_features(self.train_df)
        X_val, y_val = self.prepare_production_features(self.val_df)
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Save scaler
        with open(self.models_dir / "production_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        # Define production models with hyperparameter tuning
        models = {
            'Production Random Forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'Production Gradient Boosting': GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            ),
            'Production Elastic Net': ElasticNet(
                alpha=0.01,
                l1_ratio=0.7,
                random_state=42,
                max_iter=2000
            )
        }
        
        # Train each model
        for model_name, model in models.items():
            print(f"🤖 Training {model_name}...")
            
            try:
                # Use MultiOutputRegressor for multi-target regression
                multi_model = MultiOutputRegressor(model)
                
                # Train model
                multi_model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_train_pred = multi_model.predict(X_train_scaled)
                y_val_pred = multi_model.predict(X_val_scaled)
                
                # Calculate metrics
                result = self._calculate_metrics(model_name, y_train, y_train_pred, y_val, y_val_pred)
                self.results[model_name] = result
                self.models[model_name] = multi_model
                
                # Save model
                model_path = self.models_dir / f"{model_name.replace(' ', '_')}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(multi_model, f)
                
                print(f"   ✅ {model_name} - Val R²: {result['overall_val_metrics']['r2']:.4f}, Val MAE: {result['overall_val_metrics']['mae']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error training {model_name}: {e}")
        
        # Create ensemble model
        print("🤖 Training Ensemble Model...")
        try:
            ensemble = VotingRegressor([
                ('rf', MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))),
                ('gb', MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, random_state=42))),
                ('en', MultiOutputRegressor(ElasticNet(alpha=0.01, random_state=42)))
            ])
            
            ensemble.fit(X_train_scaled, y_train)
            y_train_pred = ensemble.predict(X_train_scaled)
            y_val_pred = ensemble.predict(X_val_scaled)
            
            result = self._calculate_metrics('Production Ensemble', y_train, y_train_pred, y_val, y_val_pred)
            self.results['Production Ensemble'] = result
            self.models['Production Ensemble'] = ensemble
            
            # Save ensemble model
            model_path = self.models_dir / "Production_Ensemble_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(ensemble, f)
            
            print(f"   ✅ Production Ensemble - Val R²: {result['overall_val_metrics']['r2']:.4f}, Val MAE: {result['overall_val_metrics']['mae']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error training ensemble: {e}")
        
        # Find best model
        if self.results:
            best_model_name = max(self.results.keys(), 
                                key=lambda x: self.results[x]['overall_val_metrics']['r2'])
            
            print(f"\n🏆 Best Model: {best_model_name}")
            print(f"   Validation R²: {self.results[best_model_name]['overall_val_metrics']['r2']:.4f}")
            print(f"   Validation MAE: {self.results[best_model_name]['overall_val_metrics']['mae']:.4f}")
            
            return best_model_name
        
        return None
    
    def _calculate_metrics(self, model_name: str, y_train: np.ndarray, y_train_pred: np.ndarray, 
                          y_val: np.ndarray, y_val_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""
        target_names = ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range', 'overall_band_score']
        
        result = {
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
            
            result['train_metrics'][target] = {
                'mse': train_mse,
                'mae': train_mae,
                'r2': train_r2
            }
            
            result['val_metrics'][target] = {
                'mse': val_mse,
                'mae': val_mae,
                'r2': val_r2
            }
        
        # Calculate overall metrics
        train_mse_overall = np.mean([result['train_metrics'][target]['mse'] for target in target_names])
        train_mae_overall = np.mean([result['train_metrics'][target]['mae'] for target in target_names])
        train_r2_overall = np.mean([result['train_metrics'][target]['r2'] for target in target_names])
        
        val_mse_overall = np.mean([result['val_metrics'][target]['mse'] for target in target_names])
        val_mae_overall = np.mean([result['val_metrics'][target]['mae'] for target in target_names])
        val_r2_overall = np.mean([result['val_metrics'][target]['r2'] for target in target_names])
        
        result['overall_train_metrics'] = {
            'mse': train_mse_overall,
            'mae': train_mae_overall,
            'r2': train_r2_overall
        }
        
        result['overall_val_metrics'] = {
            'mse': val_mse_overall,
            'mae': val_mae_overall,
            'r2': val_r2_overall
        }
        
        return result
    
    def evaluate_on_test_set(self, best_model_name: str):
        """Evaluate the best model on test set"""
        print(f"\n🧪 Evaluating {best_model_name} on Test Set")
        print("=" * 60)
        
        # Load scaler
        with open(self.models_dir / "production_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        # Load best model
        model_path = self.models_dir / f"{best_model_name.replace(' ', '_')}_model.pkl"
        with open(model_path, 'rb') as f:
            best_model = pickle.load(f)
        
        # Prepare test features
        X_test, y_test = self.prepare_production_features(self.test_df)
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
        
        with open(self.models_dir / "production_test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)
        
        return test_results

def main():
    """Main production training function"""
    data_dir = "/Users/shan/Desktop/Work/Projects/EdPrep AI/edprep-ai-prototype/data"
    models_dir = "/Users/shan/Desktop/Work/Projects/EdPrep AI/edprep-ai-prototype/models"
    
    trainer = ProductionIELTSModelTrainer(data_dir, models_dir)
    
    # Train production models
    best_model_name = trainer.train_production_models()
    
    if best_model_name:
        # Evaluate on test set
        test_results = trainer.evaluate_on_test_set(best_model_name)
        
        print("\n🎉 Production Model Training Completed!")
        print(f"🏆 Best Model: {best_model_name}")
        print(f"📁 Models saved to: {models_dir}")
        
        return trainer, best_model_name, test_results
    else:
        print("❌ No models were successfully trained")
        return None, None, None

if __name__ == "__main__":
    main()
