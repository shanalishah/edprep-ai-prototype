#!/usr/bin/env python3
"""
Local Advanced Training for EdPrep AI
Optimized for local machine training with your full dataset
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import your existing models
from models.essay_scorer import EssayScorer
from models.ml_essay_scorer import MLEssayScorer
from models.production_essay_scorer import ProductionEssayScorer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class LocalAdvancedTrainer:
    """
    Advanced training system optimized for local machine
    """
    
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize existing scorers
        self.rubric_scorer = EssayScorer()
        self.ml_scorer = MLEssayScorer()
        self.production_scorer = ProductionEssayScorer()
        
        print(f"🚀 Local Advanced Training Initialized")
        print(f"📁 Models will be saved to: {self.models_dir}")
    
    def load_dataset(self):
        """
        Load the full dataset
        """
        print("📊 Loading full dataset...")
        
        data_path = Path(__file__).parent.parent / "data" / "full_dataset.csv"
        
        try:
            df = pd.read_csv(data_path, encoding='utf-8')
            print(f"✅ Loaded {len(df)} essays")
        except Exception as e:
            print(f"⚠️ UTF-8 failed, trying latin-1: {e}")
            df = pd.read_csv(data_path, encoding='latin-1')
            print(f"✅ Loaded with latin-1: {len(df)} essays")
        
        # Clean data
        print("🧹 Cleaning dataset...")
        df = df.dropna(subset=['essay', 'overall_band_score'])
        df = df[df['overall_band_score'].between(1, 9)]
        df = df[df['essay'].str.len() > 50]  # At least 50 characters
        df = df[df['essay'].str.len() < 10000]  # Not too long
        
        print(f"✅ Cleaned dataset: {len(df)} essays")
        return df
    
    def extract_advanced_features(self, df):
        """
        Extract comprehensive features for strict scoring
        """
        print("🔧 Extracting advanced features...")
        
        features = []
        
        for idx, row in df.iterrows():
            if idx % 5000 == 0:
                print(f"  Processing essay {idx}/{len(df)}")
            
            essay = row['essay']
            prompt = row.get('prompt', '')
            
            # Basic features
            word_count = len(essay.split())
            sentence_count = len(re.split(r'[.!?]', essay))
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            paragraph_count = essay.count('\n\n') + 1
            
            # Advanced linguistic features
            complex_words = len([w for w in essay.split() if len(w) > 6])
            complexity_ratio = complex_words / word_count if word_count > 0 else 0
            
            # Cohesion features
            cohesive_markers = ['however', 'therefore', 'moreover', 'furthermore', 'consequently', 
                              'in addition', 'on the other hand', 'as a result', 'for example', 'for instance']
            cohesion_score = sum(essay.lower().count(marker) for marker in cohesive_markers)
            
            # Task-specific features
            prompt_keywords = set(re.findall(r'\b\w+\b', prompt.lower()))
            essay_keywords = set(re.findall(r'\b\w+\b', essay.lower()))
            keyword_overlap = len(prompt_keywords.intersection(essay_keywords))
            
            # Readability features (simplified)
            avg_word_length = np.mean([len(word) for word in essay.split() if word.isalpha()]) if word_count > 0 else 0
            
            # Grammar and structure features
            question_marks = essay.count('?')
            exclamation_marks = essay.count('!')
            commas = essay.count(',')
            semicolons = essay.count(';')
            colons = essay.count(':')
            
            # Essay structure analysis
            intro_indicators = ['introduction', 'firstly', 'to begin', 'initially']
            conclusion_indicators = ['conclusion', 'finally', 'in conclusion', 'to conclude', 'in summary']
            
            has_intro = any(indicator in essay.lower() for indicator in intro_indicators)
            has_conclusion = any(indicator in essay.lower() for indicator in conclusion_indicators)
            
            features.append({
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length,
                'paragraph_count': paragraph_count,
                'complexity_ratio': complexity_ratio,
                'cohesion_score': cohesion_score,
                'keyword_overlap': keyword_overlap,
                'avg_word_length': avg_word_length,
                'question_marks': question_marks,
                'exclamation_marks': exclamation_marks,
                'commas': commas,
                'semicolons': semicolons,
                'colons': colons,
                'has_intro': int(has_intro),
                'has_conclusion': int(has_conclusion),
                'essay': essay,
                'prompt': prompt
            })
        
        features_df = pd.DataFrame(features)
        print(f"✅ Extracted {len(features_df.columns)} features")
        
        return features_df
    
    def create_tfidf_features(self, essays, max_features=1000):
        """
        Create TF-IDF features for text analysis
        """
        print("📝 Creating TF-IDF features...")
        
        # Clean essays for TF-IDF
        cleaned_essays = []
        for essay in essays:
            # Remove special characters and normalize
            cleaned = re.sub(r'[^\w\s]', ' ', essay.lower())
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            cleaned_essays.append(cleaned)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=5,  # Ignore terms that appear in less than 5 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        tfidf_matrix = vectorizer.fit_transform(cleaned_essays)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        print(f"✅ Created {tfidf_df.shape[1]} TF-IDF features")
        
        return tfidf_df, vectorizer
    
    def train_ensemble_models(self, X, y, target_name="overall_band_score"):
        """
        Train ensemble of advanced models
        """
        print(f"🔧 Training ensemble models for {target_name}...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Define models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'elastic_net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42,
                max_iter=2000
            ),
            'ridge': Ridge(
                alpha=1.0,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Train model
            if name in ['elastic_net', 'ridge']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            
            # Evaluate
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            # Cross-validation
            if name in ['elastic_net', 'ridge']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            
            cv_mae = -cv_scores.mean()
            
            results[name] = {
                'model': model,
                'scaler': scaler if name in ['elastic_net', 'ridge'] else None,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'cv_mae': cv_mae
            }
            
            print(f"    ✅ {name}: MAE={mae:.3f}, R²={r2:.3f}, CV-MAE={cv_mae:.3f}")
        
        # Find best model
        best_model_name = min(results.keys(), key=lambda k: results[k]['mae'])
        print(f"🏆 Best model: {best_model_name} (MAE: {results[best_model_name]['mae']:.3f})")
        
        return results, best_model_name
    
    def create_strict_scoring_system(self, df):
        """
        Create a strict scoring system optimized for real IELTS-like assessment
        """
        print("🎯 Creating Strict Scoring System...")
        
        # Extract features
        features_df = self.extract_advanced_features(df)
        
        # Create TF-IDF features
        tfidf_df, tfidf_vectorizer = self.create_tfidf_features(features_df['essay'].tolist())
        
        # Combine features
        feature_columns = [col for col in features_df.columns if col not in ['essay', 'prompt']]
        X_basic = features_df[feature_columns]
        X_combined = pd.concat([X_basic, tfidf_df], axis=1)
        
        print(f"📊 Total features: {X_combined.shape[1]}")
        
        # Train models for each criterion
        criteria = ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range', 'overall_band_score']
        
        all_models = {}
        best_models = {}
        
        for criterion in criteria:
            if criterion in df.columns:
                print(f"\n🎯 Training models for {criterion}...")
                y = df[criterion]
                
                # Train ensemble models
                models, best_model = self.train_ensemble_models(X_combined, y, criterion)
                all_models[criterion] = models
                best_models[criterion] = best_model
        
        # Save models
        self.save_models(all_models, best_models, tfidf_vectorizer, feature_columns)
        
        return all_models, best_models, tfidf_vectorizer, feature_columns
    
    def save_models(self, all_models, best_models, tfidf_vectorizer, feature_columns):
        """
        Save all trained models
        """
        print("💾 Saving models...")
        
        # Save TF-IDF vectorizer
        joblib.dump(tfidf_vectorizer, self.models_dir / "strict_tfidf_vectorizer.pkl")
        print(f"✅ Saved TF-IDF vectorizer")
        
        # Save feature columns
        joblib.dump(feature_columns, self.models_dir / "strict_feature_columns.pkl")
        print(f"✅ Saved feature columns")
        
        # Save all models for each criterion
        for criterion, models in all_models.items():
            criterion_dir = self.models_dir / f"strict_{criterion}"
            criterion_dir.mkdir(exist_ok=True)
            
            for model_name, result in models.items():
                # Save model
                model_path = criterion_dir / f"{model_name}_model.pkl"
                joblib.dump(result['model'], model_path)
                
                # Save scaler if exists
                if result['scaler'] is not None:
                    scaler_path = criterion_dir / f"{model_name}_scaler.pkl"
                    joblib.dump(result['scaler'], scaler_path)
                
                print(f"✅ Saved {criterion}/{model_name}")
        
        # Save best model info
        joblib.dump(best_models, self.models_dir / "strict_best_models.pkl")
        print(f"✅ Saved best model selections")
        
        # Create model info file
        model_info = {
            'total_features': len(feature_columns) + tfidf_vectorizer.max_features,
            'basic_features': len(feature_columns),
            'tfidf_features': tfidf_vectorizer.max_features,
            'criteria': list(all_models.keys()),
            'best_models': best_models,
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        import json
        with open(self.models_dir / "strict_model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Saved model info")
        print(f"📁 All models saved to: {self.models_dir}")
    
    def run_training(self):
        """
        Run the complete training pipeline
        """
        print("🚀 Starting Local Advanced Training Pipeline")
        print("=" * 60)
        
        # Load dataset
        df = self.load_dataset()
        
        # Create strict scoring system
        all_models, best_models, tfidf_vectorizer, feature_columns = self.create_strict_scoring_system(df)
        
        print("\n✅ Training Complete!")
        print("=" * 60)
        print(f"📊 Trained models for {len(all_models)} criteria")
        print(f"🔧 Total features: {len(feature_columns) + tfidf_vectorizer.max_features}")
        print(f"📁 Models saved to: {self.models_dir}")
        
        # Print best models
        print("\n🏆 Best Models:")
        for criterion, best_model in best_models.items():
            mae = all_models[criterion][best_model]['mae']
            print(f"  {criterion}: {best_model} (MAE: {mae:.3f})")
        
        return all_models, best_models, tfidf_vectorizer, feature_columns

def main():
    """
    Main function to run local training
    """
    print("🎯 EdPrep AI: Local Advanced Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = LocalAdvancedTrainer()
    
    # Run training
    results = trainer.run_training()
    
    print("\n🎉 Training Complete!")
    print("Your strict scoring models are ready to use!")

if __name__ == "__main__":
    main()
