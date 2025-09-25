"""
Comprehensive Data Preprocessing Pipeline for IELTS Writing Datasets
This pipeline handles all different data formats and creates a unified training dataset
"""

import pandas as pd
import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class IELTSDataPreprocessor:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load analysis results
        self.analysis_file = self.output_dir / "data_analysis.json"
        self.recommendations_file = self.output_dir / "training_recommendations.json"
        
        if self.analysis_file.exists():
            with open(self.analysis_file, 'r') as f:
                self.file_analysis = json.load(f)
        else:
            self.file_analysis = {}
        
        if self.recommendations_file.exists():
            with open(self.recommendations_file, 'r') as f:
                self.recommendations = json.load(f)
        else:
            self.recommendations = {}
    
    def extract_scores_from_feedback(self, feedback_text: str) -> Dict[str, float]:
        """Extract scores from feedback text using multiple patterns"""
        scores = {
            'task_achievement': None,
            'coherence_cohesion': None,
            'lexical_resource': None,
            'grammatical_range': None,
            'overall_band_score': None
        }
        
        if not feedback_text or pd.isna(feedback_text):
            return scores
        
        feedback_lower = feedback_text.lower()
        
        # Multiple patterns for each criterion
        patterns = {
            'task_achievement': [
                r'task achievement[:\s]*(\d+\.?\d*)',
                r'task response[:\s]*(\d+\.?\d*)',
                r'task[:\s]*(\d+\.?\d*)',
                r'achievement[:\s]*(\d+\.?\d*)'
            ],
            'coherence_cohesion': [
                r'coherence and cohesion[:\s]*(\d+\.?\d*)',
                r'coherence[:\s]*(\d+\.?\d*)',
                r'cohesion[:\s]*(\d+\.?\d*)'
            ],
            'lexical_resource': [
                r'lexical resource[:\s]*(\d+\.?\d*)',
                r'vocabulary[:\s]*(\d+\.?\d*)',
                r'lexical[:\s]*(\d+\.?\d*)'
            ],
            'grammatical_range': [
                r'grammatical range[:\s]*(\d+\.?\d*)',
                r'grammar[:\s]*(\d+\.?\d*)',
                r'grammatical[:\s]*(\d+\.?\d*)'
            ],
            'overall_band_score': [
                r'overall band score[:\s]*(\d+\.?\d*)',
                r'overall[:\s]*(\d+\.?\d*)',
                r'band score[:\s]*(\d+\.?\d*)',
                r'band[:\s]*(\d+\.?\d*)'
            ]
        }
        
        for criterion, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, feedback_lower)
                if match:
                    try:
                        score = float(match.group(1))
                        if 1.0 <= score <= 9.0:  # Valid IELTS band score range
                            scores[criterion] = score
                            break
                    except ValueError:
                        continue
        
        return scores
    
    def clean_essay_text(self, text: str) -> str:
        """Clean and normalize essay text"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def process_csv_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a CSV file and extract essays with scores"""
        try:
            df = pd.read_csv(file_path)
            processed_data = []
            
            # Get column mappings from analysis
            filename = file_path.name
            if filename in self.file_analysis:
                analysis = self.file_analysis[filename]
                essay_cols = analysis.get('essay_columns', [])
                prompt_cols = analysis.get('prompt_columns', [])
                score_cols = analysis.get('score_columns', [])
                feedback_cols = analysis.get('feedback_columns', [])
            else:
                # Fallback column detection
                essay_cols = [col for col in df.columns if 'essay' in col.lower() or 'text' in col.lower()]
                prompt_cols = [col for col in df.columns if 'prompt' in col.lower() or 'question' in col.lower()]
                score_cols = [col for col in df.columns if 'band' in col.lower() or 'score' in col.lower()]
                feedback_cols = [col for col in df.columns if 'evaluation' in col.lower() or 'feedback' in col.lower()]
            
            for idx, row in df.iterrows():
                # Extract essay
                essay = ""
                for col in essay_cols:
                    if col in df.columns and pd.notna(row[col]):
                        essay = self.clean_essay_text(row[col])
                        break
                
                if not essay or len(essay.split()) < 50:  # Skip very short essays
                    continue
                
                # Extract prompt
                prompt = ""
                for col in prompt_cols:
                    if col in df.columns and pd.notna(row[col]):
                        prompt = str(row[col]).strip()
                        break
                
                # Extract scores
                scores = {}
                
                # Try to get scores from dedicated score columns
                for col in score_cols:
                    if col in df.columns and pd.notna(row[col]):
                        try:
                            score = float(row[col])
                            if 1.0 <= score <= 9.0:
                                if 'overall' in col.lower() or 'band' in col.lower():
                                    scores['overall_band_score'] = score
                                elif 'task' in col.lower():
                                    scores['task_achievement'] = score
                                elif 'coherence' in col.lower():
                                    scores['coherence_cohesion'] = score
                                elif 'lexical' in col.lower():
                                    scores['lexical_resource'] = score
                                elif 'grammatical' in col.lower():
                                    scores['grammatical_range'] = score
                        except (ValueError, TypeError):
                            continue
                
                # If no scores from columns, try to extract from feedback
                if not scores and feedback_cols:
                    for col in feedback_cols:
                        if col in df.columns and pd.notna(row[col]):
                            feedback_scores = self.extract_scores_from_feedback(str(row[col]))
                            scores.update({k: v for k, v in feedback_scores.items() if v is not None})
                            break
                
                # Only include if we have at least one score
                if scores:
                    processed_data.append({
                        'essay': essay,
                        'prompt': prompt,
                        'scores': scores,
                        'source_file': filename,
                        'word_count': len(essay.split())
                    })
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            return []
    
    def process_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a JSONL file and extract essays with scores"""
        try:
            processed_data = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract essay
                        essay = ""
                        if 'input' in data:
                            essay = self.clean_essay_text(data['input'])
                        elif 'essay' in data:
                            essay = self.clean_essay_text(data['essay'])
                        
                        if not essay or len(essay.split()) < 50:
                            continue
                        
                        # Extract prompt
                        prompt = ""
                        if 'instruction' in data:
                            prompt = str(data['instruction']).strip()
                        elif 'prompt' in data:
                            prompt = str(data['prompt']).strip()
                        
                        # Extract scores from feedback
                        scores = {}
                        if 'output' in data:
                            feedback_scores = self.extract_scores_from_feedback(str(data['output']))
                            scores.update({k: v for k, v in feedback_scores.items() if v is not None})
                        elif 'feedback' in data:
                            if isinstance(data['feedback'], dict):
                                scores = {k: v for k, v in data['feedback'].items() 
                                        if isinstance(v, (int, float)) and 1.0 <= v <= 9.0}
                            else:
                                feedback_scores = self.extract_scores_from_feedback(str(data['feedback']))
                                scores.update({k: v for k, v in feedback_scores.items() if v is not None})
                        
                        if scores:
                            processed_data.append({
                                'essay': essay,
                                'prompt': prompt,
                                'scores': scores,
                                'source_file': file_path.name,
                                'word_count': len(essay.split())
                            })
                    
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error processing line {line_num} in {file_path.name}: {e}")
                        continue
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            return []
    
    def process_excel_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process an Excel file and extract essays with scores"""
        try:
            df = pd.read_excel(file_path)
            # Use the same logic as CSV processing
            return self.process_csv_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            return []
    
    def process_all_files(self) -> List[Dict[str, Any]]:
        """Process all data files and create unified dataset"""
        print("🔄 Processing all data files...")
        print("=" * 60)
        
        all_data = []
        
        # Get top recommended files first
        top_files = []
        if 'top_files' in self.recommendations:
            for file_info in self.recommendations['top_files']:
                filename = file_info['filename']
                file_path = self.data_dir / filename
                if file_path.exists():
                    top_files.append((file_path, file_info['score']))
        
        # Sort by quality score
        top_files.sort(key=lambda x: x[1], reverse=True)
        
        print(f"📊 Processing {len(top_files)} top-quality files first...")
        
        for file_path, score in top_files:
            print(f"📄 Processing: {file_path.name} (Score: {score:.1f})")
            
            if file_path.suffix == '.csv':
                data = self.process_csv_file(file_path)
            elif file_path.suffix == '.jsonl':
                data = self.process_jsonl_file(file_path)
            elif file_path.suffix == '.xlsx':
                data = self.process_excel_file(file_path)
            else:
                continue
            
            print(f"   ✅ Extracted {len(data)} essays with scores")
            all_data.extend(data)
        
        # Process remaining files
        processed_files = {f[0].name for f in top_files}
        remaining_files = [f for f in self.data_dir.glob('*') 
                          if f.suffix in ['.csv', '.jsonl', '.xlsx'] 
                          and f.name not in processed_files]
        
        print(f"\n📄 Processing {len(remaining_files)} remaining files...")
        
        for file_path in remaining_files:
            print(f"📄 Processing: {file_path.name}")
            
            if file_path.suffix == '.csv':
                data = self.process_csv_file(file_path)
            elif file_path.suffix == '.jsonl':
                data = self.process_jsonl_file(file_path)
            elif file_path.suffix == '.xlsx':
                data = self.process_excel_file(file_path)
            else:
                continue
            
            if data:
                print(f"   ✅ Extracted {len(data)} essays with scores")
                all_data.extend(data)
            else:
                print(f"   ⚠️  No valid data extracted")
        
        print(f"\n🎯 Total essays extracted: {len(all_data)}")
        return all_data
    
    def create_training_dataset(self, all_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a unified training dataset"""
        print("\n🔧 Creating unified training dataset...")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Expand scores into separate columns
        score_columns = ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range', 'overall_band_score']
        
        for col in score_columns:
            df[col] = df['scores'].apply(lambda x: x.get(col) if isinstance(x, dict) else None)
        
        # Drop the original scores column
        df = df.drop('scores', axis=1)
        
        # Filter out rows with no scores
        df = df.dropna(subset=['overall_band_score'])
        
        # Clean and validate data
        df = df[df['word_count'] >= 100]  # Minimum word count
        df = df[df['word_count'] <= 1000]  # Maximum word count
        
        # Fill missing scores with median values
        for col in score_columns:
            if col in df.columns:
                median_score = df[col].median()
                df[col] = df[col].fillna(median_score)
        
        # Create additional features
        df['essay_length'] = df['essay'].str.len()
        df['avg_word_length'] = df['essay'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x else 0)
        df['sentence_count'] = df['essay'].apply(lambda x: len(re.split(r'[.!?]+', x)) if x else 0)
        df['paragraph_count'] = df['essay'].apply(lambda x: len([p for p in x.split('\n\n') if p.strip()]) if x else 0)
        
        # Add complexity features
        df['complex_sentences'] = df['essay'].apply(lambda x: len(re.findall(r'\b(because|although|while|whereas|if|when|since|as)\b', x.lower())) if x else 0)
        df['linking_words'] = df['essay'].apply(lambda x: len(re.findall(r'\b(however|therefore|moreover|furthermore|additionally|on the other hand|in contrast|similarly|likewise|firstly|secondly|finally|in addition|as a result)\b', x.lower())) if x else 0)
        
        print(f"✅ Final dataset: {len(df)} essays")
        print(f"📊 Score distribution:")
        for col in score_columns:
            if col in df.columns:
                print(f"   {col}: {df[col].mean():.2f} ± {df[col].std():.2f}")
        
        return df
    
    def save_datasets(self, df: pd.DataFrame):
        """Save the processed datasets"""
        print("\n💾 Saving datasets...")
        
        # Split into train/validation/test
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        # Save datasets
        train_df.to_csv(self.output_dir / "train_dataset.csv", index=False)
        val_df.to_csv(self.output_dir / "val_dataset.csv", index=False)
        test_df.to_csv(self.output_dir / "test_dataset.csv", index=False)
        df.to_csv(self.output_dir / "full_dataset.csv", index=False)
        
        print(f"✅ Saved datasets:")
        print(f"   📄 train_dataset.csv: {len(train_df)} essays")
        print(f"   📄 val_dataset.csv: {len(val_df)} essays")
        print(f"   📄 test_dataset.csv: {len(test_df)} essays")
        print(f"   📄 full_dataset.csv: {len(df)} essays")
        
        # Save dataset statistics
        stats = {
            'total_essays': len(df),
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'score_statistics': {
                col: {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
                for col in ['task_achievement', 'coherence_cohesion', 'lexical_resource', 'grammatical_range', 'overall_band_score']
                if col in df.columns
            },
            'word_count_stats': {
                'mean': float(df['word_count'].mean()),
                'std': float(df['word_count'].std()),
                'min': int(df['word_count'].min()),
                'max': int(df['word_count'].max())
            }
        }
        
        with open(self.output_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"   📄 dataset_statistics.json: Dataset statistics")
        
        return train_df, val_df, test_df

def main():
    """Main preprocessing function"""
    data_dir = "/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Writing/data_writing"
    output_dir = "/Users/shan/Desktop/Work/Projects/EdPrep AI/edprep-ai-prototype/data"
    
    preprocessor = IELTSDataPreprocessor(data_dir, output_dir)
    
    # Process all files
    all_data = preprocessor.process_all_files()
    
    if not all_data:
        print("❌ No data extracted. Please check your data files.")
        return
    
    # Create unified dataset
    df = preprocessor.create_training_dataset(all_data)
    
    # Save datasets
    train_df, val_df, test_df = preprocessor.save_datasets(df)
    
    print("\n🎉 Data preprocessing completed successfully!")
    print(f"📊 Ready for model training with {len(df)} essays")
    
    return df, train_df, val_df, test_df

if __name__ == "__main__":
    main()
