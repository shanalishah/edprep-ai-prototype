"""
Data preprocessing script for IELTS writing datasets
This script processes the existing IELTS data for model training
"""

import pandas as pd
import json
import re
from typing import Dict, List, Tuple
import numpy as np

def load_ielts_data(data_path: str) -> pd.DataFrame:
    """
    Load IELTS dataset from JSONL format
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    return pd.DataFrame(data)

def extract_scores_from_feedback(feedback_text: str) -> Dict[str, float]:
    """
    Extract scores from feedback text using regex patterns
    """
    scores = {
        'task_achievement': 5.0,
        'coherence_cohesion': 5.0,
        'lexical_resource': 5.0,
        'grammatical_range': 5.0,
        'overall_band_score': 5.0
    }
    
    # Patterns to extract scores
    patterns = {
        'task_achievement': r'Task Achievement[:\s]*(\d+\.?\d*)',
        'coherence_cohesion': r'Coherence and Cohesion[:\s]*(\d+\.?\d*)',
        'lexical_resource': r'Lexical Resource[:\s]*(\d+\.?\d*)',
        'grammatical_range': r'Grammatical Range[:\s]*(\d+\.?\d*)',
        'overall_band_score': r'Overall Band Score[:\s]*(\d+\.?\d*)'
    }
    
    for criterion, pattern in patterns.items():
        match = re.search(pattern, feedback_text, re.IGNORECASE)
        if match:
            try:
                scores[criterion] = float(match.group(1))
            except ValueError:
                continue
    
    return scores

def clean_essay_text(text: str) -> str:
    """
    Clean and normalize essay text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()

def preprocess_ielts_dataset(input_path: str, output_path: str):
    """
    Main preprocessing function
    """
    print("Loading IELTS dataset...")
    df = load_ielts_data(input_path)
    
    print(f"Loaded {len(df)} essays")
    
    # Extract scores from feedback
    print("Extracting scores from feedback...")
    score_data = []
    for idx, row in df.iterrows():
        if 'output' in row:
            scores = extract_scores_from_feedback(row['output'])
            score_data.append(scores)
        else:
            score_data.append({
                'task_achievement': 5.0,
                'coherence_cohesion': 5.0,
                'lexical_resource': 5.0,
                'grammatical_range': 5.0,
                'overall_band_score': 5.0
            })
    
    # Add scores to dataframe
    scores_df = pd.DataFrame(score_data)
    df = pd.concat([df, scores_df], axis=1)
    
    # Clean essay text
    print("Cleaning essay text...")
    if 'input' in df.columns:
        df['essay_clean'] = df['input'].apply(clean_essay_text)
    
    # Filter out essays that are too short or too long
    df['word_count'] = df['essay_clean'].apply(lambda x: len(x.split()) if x else 0)
    df = df[(df['word_count'] >= 100) & (df['word_count'] <= 1000)]
    
    print(f"After filtering: {len(df)} essays")
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Average word count: {df['word_count'].mean():.1f}")
    print(f"Average overall score: {df['overall_band_score'].mean():.2f}")
    print(f"Score distribution:")
    for criterion in ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range']:
        print(f"  {criterion}: {df[criterion].mean():.2f} ± {df[criterion].std():.2f}")

def create_training_data():
    """
    Create training data from the existing IELTS datasets
    """
    # Paths to your existing data
    data_paths = [
        "/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Writing/data_writing/IELTS_dataset_5000_v2.jsonl",
        "/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Writing/data_writing/ielts_essays_sample.jsonl"
    ]
    
    output_path = "/Users/shan/Desktop/Work/Projects/EdPrep AI/edprep-ai-prototype/data/processed_ielts_data.csv"
    
    for data_path in data_paths:
        try:
            print(f"Processing {data_path}...")
            preprocess_ielts_dataset(data_path, output_path)
        except FileNotFoundError:
            print(f"File not found: {data_path}")
        except Exception as e:
            print(f"Error processing {data_path}: {e}")

if __name__ == "__main__":
    create_training_data()

