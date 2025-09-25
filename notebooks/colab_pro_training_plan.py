#!/usr/bin/env python3
"""
Google Colab Pro Training Plan for EdPrep AI
Advanced training pipeline for strict IELTS-like assessment
"""

# This script is designed to run on Google Colab Pro
# Copy this to Colab and run with your full dataset

import pandas as pd
import numpy as np
import torch
import transformers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class ColabProTrainingPipeline:
    """
    Advanced training pipeline for Google Colab Pro
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Initialize transformers
        self.tokenizer = None
        self.model = None
        
    def load_and_prepare_data(self, data_path):
        """
        Load and prepare the full dataset with robust CSV handling
        """
        print("📊 Loading full dataset...")
        
        try:
            # Try to load with robust settings
            df = pd.read_csv(data_path, 
                           encoding='utf-8',
                           on_bad_lines='skip',
                           quoting=1)  # QUOTE_ALL
            print(f"✅ Loaded {len(df)} essays")
        except Exception as e:
            print(f"⚠️ CSV loading error: {e}")
            print("🔧 Trying alternative loading methods...")
            
            try:
                # Try with different encoding
                df = pd.read_csv(data_path, 
                               encoding='latin-1',
                               on_bad_lines='skip')
                print(f"✅ Loaded with latin-1 encoding: {len(df)} essays")
            except Exception as e2:
                print(f"⚠️ Latin-1 failed: {e2}")
                print("🔧 Trying chunk-based loading...")
                
                # Load in chunks
                chunk_list = []
                chunk_size = 10000
                
                for chunk in pd.read_csv(data_path, 
                                       chunksize=chunk_size,
                                       on_bad_lines='skip',
                                       encoding='utf-8'):
                    chunk_list.append(chunk)
                
                df = pd.concat(chunk_list, ignore_index=True)
                print(f"✅ Loaded in chunks: {len(df)} essays")
        
        # Clean and prepare data
        print("🧹 Cleaning dataset...")
        df = df.dropna(subset=['essay', 'overall_band_score'])
        df = df[df['overall_band_score'].between(1, 9)]
        
        # Remove essays that are too short or too long
        df = df[df['essay'].str.len() > 50]  # At least 50 characters
        df = df[df['essay'].str.len() < 10000]  # Not too long
        
        print(f"✅ Cleaned dataset: {len(df)} essays")
        return df
    
    def extract_advanced_features(self, df):
        """
        Extract advanced features for strict scoring
        """
        print("🔧 Extracting advanced features...")
        
        features = []
        
        for idx, row in df.iterrows():
            essay = row['essay']
            prompt = row.get('prompt', '')
            
            # Basic features
            word_count = len(essay.split())
            sentence_count = len(essay.split('.'))
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Advanced linguistic features
            complex_words = len([w for w in essay.split() if len(w) > 6])
            complexity_ratio = complex_words / word_count if word_count > 0 else 0
            
            # Cohesion features
            cohesive_markers = ['however', 'therefore', 'moreover', 'furthermore', 'consequently']
            cohesion_score = sum(essay.lower().count(marker) for marker in cohesive_markers)
            
            # Task-specific features
            prompt_keywords = set(prompt.lower().split())
            essay_keywords = set(essay.lower().split())
            keyword_overlap = len(prompt_keywords.intersection(essay_keywords))
            
            # Readability features
            try:
                from textstat import flesch_reading_ease, gunning_fog
                readability = flesch_reading_ease(essay)
                fog_index = gunning_fog(essay)
            except:
                readability = 0
                fog_index = 0
            
            features.append({
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length,
                'complexity_ratio': complexity_ratio,
                'cohesion_score': cohesion_score,
                'keyword_overlap': keyword_overlap,
                'readability': readability,
                'fog_index': fog_index,
                'essay': essay,
                'prompt': prompt
            })
        
        features_df = pd.DataFrame(features)
        print(f"✅ Extracted {len(features_df.columns)} features")
        
        return features_df
    
    def train_transformer_model(self, df, target_column='overall_band_score'):
        """
        Train a transformer-based model for strict scoring
        """
        print("🤖 Training Transformer Model...")
        
        # Initialize tokenizer and model
        model_name = 'distilbert-base-uncased'  # Faster than BERT
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=1  # Regression task
        )
        
        # Prepare data
        essays = df['essay'].tolist()
        scores = df[target_column].tolist()
        
        # Tokenize
        encodings = self.tokenizer(
            essays, 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_tensors='pt'
        )
        
        # Create dataset
        class EssayDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
                return item
            
            def __len__(self):
                return len(self.labels)
        
        dataset = EssayDataset(encodings, scores)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Training arguments
        training_args = transformers.TrainingArguments(
            output_dir='./transformer_model',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="steps",  # Updated parameter name
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
        )
        
        # Trainer
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        trainer.train()
        
        print("✅ Transformer model trained successfully")
        return trainer
    
    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation
        """
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        
        return {
            'mae': mae,
            'r2': r2
        }
    
    def train_ensemble_models(self, X, y):
        """
        Train ensemble of traditional ML models
        """
        print("🔧 Training Ensemble Models...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'random_forest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42),
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'r2': r2
            }
            
            print(f"✅ {name}: MAE={mae:.3f}, R²={r2:.3f}")
        
        return results
    
    def create_strict_scoring_system(self, df):
        """
        Create a strict scoring system optimized for real IELTS-like assessment
        """
        print("🎯 Creating Strict Scoring System...")
        
        # Extract features
        features_df = self.extract_advanced_features(df)
        
        # Prepare features for ML
        feature_columns = [col for col in features_df.columns if col not in ['essay', 'prompt']]
        X = features_df[feature_columns]
        y = df['overall_band_score']
        
        # Train ensemble models
        ensemble_results = self.train_ensemble_models(X, y)
        
        # Train transformer model
        transformer_trainer = self.train_transformer_model(df)
        
        # Create strict scoring function
        def strict_score(essay, prompt, models, transformer_model):
            """
            Strict scoring function that penalizes low-quality essays more
            """
            # Extract features
            word_count = len(essay.split())
            sentence_count = len(essay.split('.'))
            
            # Basic quality checks
            if word_count < 250:  # Task 2 minimum
                return 4.0  # Strict penalty for short essays
            
            if sentence_count < 5:
                return 4.5  # Penalty for poor structure
            
            # Get ML predictions
            ml_scores = []
            for name, result in models.items():
                # Extract features for this essay
                features = self.extract_advanced_features(pd.DataFrame([{
                    'essay': essay,
                    'prompt': prompt
                }]))
                feature_values = features[feature_columns].iloc[0].values.reshape(1, -1)
                score = result['model'].predict(feature_values)[0]
                ml_scores.append(score)
            
            # Get transformer prediction
            transformer_score = self.get_transformer_prediction(essay, transformer_model)
            
            # Combine predictions with strict weighting
            combined_score = np.mean(ml_scores + [transformer_score])
            
            # Apply strict adjustments
            if word_count < 300:
                combined_score -= 0.5  # Penalty for short essays
            
            if combined_score < 5.0:
                combined_score -= 0.3  # Additional penalty for low scores
            
            # Round to nearest 0.5
            return round(combined_score * 2) / 2
        
        return strict_score, ensemble_results, transformer_trainer
    
    def get_transformer_prediction(self, essay, model):
        """
        Get prediction from transformer model
        """
        inputs = self.tokenizer(essay, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.logits.item()
    
    def save_models(self, models, transformer_model, output_dir='./strict_models'):
        """
        Save trained models
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save ensemble models
        for name, result in models.items():
            joblib.dump(result['model'], f'{output_dir}/{name}_strict.pkl')
        
        # Save transformer model
        transformer_model.save_pretrained(f'{output_dir}/transformer_strict')
        self.tokenizer.save_pretrained(f'{output_dir}/transformer_strict')
        
        print(f"✅ Models saved to {output_dir}")

def main():
    """
    Main training pipeline for Google Colab Pro
    """
    print("🚀 EdPrep AI: Google Colab Pro Training Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ColabProTrainingPipeline()
    
    # Load data (replace with your actual data path)
    # Option 1: Upload directly to Colab
    data_path = '/content/full_dataset.csv'  # Upload your data to Colab
    
    # Option 2: Use Google Drive (uncomment if using Drive)
    # data_path = '/content/drive/MyDrive/full_dataset.csv'
    
    # Check if file exists
    import os
    if not os.path.exists(data_path):
        print(f"❌ File not found at: {data_path}")
        print("Available files in /content/:")
        print(os.listdir('/content/'))
        print("\nPlease upload full_dataset.csv to Google Colab using the folder icon")
        return
    df = pipeline.load_and_prepare_data(data_path)
    
    # Create strict scoring system
    strict_scorer, models, transformer = pipeline.create_strict_scoring_system(df)
    
    # Save models
    pipeline.save_models(models, transformer)
    
    print("✅ Training Complete!")
    print("Download the models and integrate them into your local system")

if __name__ == "__main__":
    main()
