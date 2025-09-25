#!/usr/bin/env python3
"""
Deep Learning Training for EdPrep AI with Progress Tracking
Includes transformer models and neural networks with real-time progress bars
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

# Progress tracking
from tqdm import tqdm
import time

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import transformers
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    from transformers import TrainingArguments, Trainer
    TORCH_AVAILABLE = True
    print("✅ PyTorch and Transformers available")
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"⚠️ PyTorch not available: {e}")
    print("Install with: pip install torch transformers")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class ProgressTracker:
    """
    Custom progress tracker for training
    """
    def __init__(self, total_steps, description="Training"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.pbar = tqdm(total=total_steps, desc=description, unit="step")
    
    def update(self, step=1, **kwargs):
        self.current_step += step
        self.pbar.update(step)
        
        # Update description with additional info
        if kwargs:
            info = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
            self.pbar.set_postfix_str(info)
    
    def close(self):
        self.pbar.close()
        elapsed = time.time() - self.start_time
        print(f"✅ {self.description} completed in {elapsed:.1f} seconds")

class EssayDataset(Dataset):
    """
    Custom dataset for essay scoring
    """
    def __init__(self, essays, scores, tokenizer, max_length=512):
        self.essays = essays
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.essays)
    
    def __getitem__(self, idx):
        essay = str(self.essays[idx])
        score = float(self.scores[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            essay,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(score, dtype=torch.float)
        }

class DeepLearningTrainer:
    """
    Deep learning trainer with progress tracking
    """
    
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Check if deep learning is available
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and Transformers are required for deep learning training")
        
        print(f"🚀 Deep Learning Training Initialized")
        print(f"📁 Models will be saved to: {self.models_dir}")
        print(f"🔥 Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    def load_dataset(self):
        """
        Load the full dataset with progress tracking
        """
        print("📊 Loading full dataset...")
        
        data_path = Path(__file__).parent.parent / "data" / "full_dataset.csv"
        
        # Progress bar for loading
        with tqdm(desc="Loading dataset", unit="rows") as pbar:
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
                pbar.update(len(df))
                print(f"✅ Loaded {len(df)} essays")
            except Exception as e:
                print(f"⚠️ UTF-8 failed, trying latin-1: {e}")
                df = pd.read_csv(data_path, encoding='latin-1')
                pbar.update(len(df))
                print(f"✅ Loaded with latin-1: {len(df)} essays")
        
        # Clean data with progress
        print("🧹 Cleaning dataset...")
        with tqdm(desc="Cleaning data", total=4) as pbar:
            df = df.dropna(subset=['essay', 'overall_band_score'])
            pbar.update(1)
            
            df = df[df['overall_band_score'].between(1, 9)]
            pbar.update(1)
            
            df = df[df['essay'].str.len() > 50]
            pbar.update(1)
            
            df = df[df['essay'].str.len() < 10000]
            pbar.update(1)
        
        print(f"✅ Cleaned dataset: {len(df)} essays")
        return df
    
    def train_transformer_models(self, df):
        """
        Train transformer models with progress tracking
        """
        print("🤖 Training Transformer Models...")
        
        # Model configurations
        models_config = [
            {
                'name': 'distilbert',
                'model_name': 'distilbert-base-uncased',
                'epochs': 2,
                'batch_size': 16
            },
            {
                'name': 'roberta',
                'model_name': 'roberta-base',
                'epochs': 2,
                'batch_size': 8
            }
        ]
        
        transformer_results = {}
        
        for config in models_config:
            print(f"\n🔄 Training {config['name']}...")
            
            try:
                # Initialize tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
                model = AutoModelForSequenceClassification.from_pretrained(
                    config['model_name'],
                    num_labels=1
                )
                
                # Prepare data
                essays = df['essay'].tolist()
                scores = df['overall_band_score'].tolist()
                
                # Split data
                train_essays, val_essays, train_scores, val_scores = train_test_split(
                    essays, scores, test_size=0.2, random_state=42
                )
                
                # Create datasets
                train_dataset = EssayDataset(train_essays, train_scores, tokenizer)
                val_dataset = EssayDataset(val_essays, val_scores, tokenizer)
                
                # Training arguments
                training_args = TrainingArguments(
                    output_dir=f'./{config["name"]}_model',
                    num_train_epochs=config['epochs'],
                    per_device_train_batch_size=config['batch_size'],
                    per_device_eval_batch_size=config['batch_size'],
                    warmup_steps=100,
                    weight_decay=0.01,
                    logging_dir=f'./{config["name"]}_logs',
                    logging_steps=50,
                    eval_strategy="steps",
                    eval_steps=200,
                    save_steps=500,
                    load_best_model_at_end=True,
                    report_to=None,  # Disable wandb
                )
                
                # Custom trainer with progress tracking
                class ProgressTrainer(Trainer):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.progress_tracker = None
                    
                    def train(self, *args, **kwargs):
                        total_steps = self.num_train_epochs * len(self.get_train_dataloader())
                        self.progress_tracker = ProgressTracker(total_steps, f"Training {config['name']}")
                        return super().train(*args, **kwargs)
                    
                    def training_step(self, model, inputs):
                        result = super().training_step(model, inputs)
                        if self.progress_tracker:
                            self.progress_tracker.update(1, loss=f"{result['loss']:.3f}")
                        return result
                
                # Initialize trainer
                trainer = ProgressTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=self.compute_metrics,
                )
                
                # Train model
                trainer.train()
                
                # Evaluate
                eval_results = trainer.evaluate()
                
                # Save model
                model_path = self.models_dir / f"transformer_{config['name']}"
                trainer.save_model(str(model_path))
                tokenizer.save_pretrained(str(model_path))
                
                transformer_results[config['name']] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'eval_results': eval_results,
                    'model_path': model_path
                }
                
                print(f"✅ {config['name']} training completed")
                print(f"📊 Evaluation results: {eval_results}")
                
            except Exception as e:
                print(f"❌ Error training {config['name']}: {e}")
                continue
        
        return transformer_results
    
    def train_neural_networks(self, df):
        """
        Train neural network models with progress tracking
        """
        print("🧠 Training Neural Networks...")
        
        # Extract features for neural networks
        print("🔧 Extracting features for neural networks...")
        features_df = self.extract_features_for_nn(df)
        
        # Prepare data
        X = features_df.drop(['essay', 'overall_band_score'], axis=1).values
        y = df['overall_band_score'].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Neural network models
        nn_models = {
            'simple_nn': self.create_simple_nn(X_train.shape[1]),
            'deep_nn': self.create_deep_nn(X_train.shape[1]),
            'wide_nn': self.create_wide_nn(X_train.shape[1])
        }
        
        nn_results = {}
        
        for name, model in nn_models.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Training setup
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Training loop with progress tracking
                epochs = 50
                batch_size = 32
                
                progress_tracker = ProgressTracker(epochs, f"Training {name}")
                
                train_losses = []
                val_losses = []
                
                for epoch in range(epochs):
                    # Training
                    model.train()
                    train_loss = 0.0
                    
                    for i in range(0, len(X_train_tensor), batch_size):
                        batch_X = X_train_tensor[i:i+batch_size]
                        batch_y = y_train_tensor[i:i+batch_size]
                        
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    train_losses.append(train_loss / len(X_train_tensor))
                    val_losses.append(val_loss)
                    
                    # Update progress
                    progress_tracker.update(1, 
                                          train_loss=f"{train_losses[-1]:.3f}",
                                          val_loss=f"{val_losses[-1]:.3f}")
                
                progress_tracker.close()
                
                # Final evaluation
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_val_tensor).numpy().flatten()
                
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                # Save model
                model_path = self.models_dir / f"neural_network_{name}.pkl"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler': scaler,
                    'feature_columns': features_df.columns.tolist()
                }, model_path)
                
                nn_results[name] = {
                    'model': model,
                    'scaler': scaler,
                    'mae': mae,
                    'r2': r2,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'model_path': model_path
                }
                
                print(f"✅ {name} training completed")
                print(f"📊 MAE: {mae:.3f}, R²: {r2:.3f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue
        
        return nn_results
    
    def extract_features_for_nn(self, df):
        """
        Extract features for neural network training
        """
        features = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            essay = row['essay']
            
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
            
            features.append({
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length,
                'complexity_ratio': complexity_ratio,
                'cohesion_score': cohesion_score,
                'essay': essay,
                'overall_band_score': row['overall_band_score']
            })
        
        return pd.DataFrame(features)
    
    def create_simple_nn(self, input_size):
        """
        Create a simple neural network
        """
        return nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def create_deep_nn(self, input_size):
        """
        Create a deep neural network
        """
        return nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    
    def create_wide_nn(self, input_size):
        """
        Create a wide neural network
        """
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def compute_metrics(self, eval_pred):
        """
        Compute metrics for transformer evaluation
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
    
    def run_deep_learning_training(self):
        """
        Run the complete deep learning training pipeline
        """
        print("🚀 Starting Deep Learning Training Pipeline")
        print("=" * 60)
        
        # Load dataset
        df = self.load_dataset()
        
        # Train transformer models
        transformer_results = self.train_transformer_models(df)
        
        # Train neural networks
        nn_results = self.train_neural_networks(df)
        
        # Save results
        self.save_deep_learning_results(transformer_results, nn_results)
        
        print("\n✅ Deep Learning Training Complete!")
        print("=" * 60)
        print(f"🤖 Trained {len(transformer_results)} transformer models")
        print(f"🧠 Trained {len(nn_results)} neural network models")
        print(f"📁 Models saved to: {self.models_dir}")
        
        return transformer_results, nn_results
    
    def save_deep_learning_results(self, transformer_results, nn_results):
        """
        Save deep learning training results
        """
        results = {
            'transformer_models': {
                name: {
                    'eval_results': result['eval_results'],
                    'model_path': str(result['model_path'])
                }
                for name, result in transformer_results.items()
            },
            'neural_network_models': {
                name: {
                    'mae': result['mae'],
                    'r2': result['r2'],
                    'model_path': str(result['model_path'])
                }
                for name, result in nn_results.items()
            },
            'training_timestamp': pd.Timestamp.now().isoformat()
        }
        
        import json
        with open(self.models_dir / "deep_learning_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Deep learning results saved")

def main():
    """
    Main function to run deep learning training
    """
    print("🎯 EdPrep AI: Deep Learning Training with Progress Tracking")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch and Transformers are required")
        print("Install with: pip install torch transformers")
        return
    
    # Initialize trainer
    trainer = DeepLearningTrainer()
    
    # Run training
    results = trainer.run_deep_learning_training()
    
    print("\n🎉 Deep Learning Training Complete!")
    print("Your advanced AI models are ready to use!")

if __name__ == "__main__":
    main()
