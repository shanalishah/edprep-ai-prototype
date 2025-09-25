"""
Comprehensive Data Analysis Script for IELTS Writing Datasets
This script analyzes all data files to understand their formats and structures
"""

import pandas as pd
import json
import os
import glob
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class IELTSDataAnalyzer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.file_analysis = {}
        self.unified_data = []
        
    def analyze_all_files(self):
        """Analyze all data files in the directory"""
        print("🔍 Analyzing IELTS Writing Data Files...")
        print("=" * 60)
        
        # Find all data files
        file_patterns = ['*.csv', '*.json', '*.jsonl', '*.xlsx']
        all_files = []
        for pattern in file_patterns:
            all_files.extend(self.data_dir.glob(pattern))
        
        print(f"Found {len(all_files)} data files")
        print()
        
        for file_path in all_files:
            print(f"📄 Analyzing: {file_path.name}")
            try:
                analysis = self._analyze_single_file(file_path)
                self.file_analysis[file_path.name] = analysis
                self._print_file_summary(file_path.name, analysis)
            except Exception as e:
                print(f"❌ Error analyzing {file_path.name}: {e}")
            print("-" * 40)
        
        return self.file_analysis
    
    def _analyze_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single data file"""
        analysis = {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'extension': file_path.suffix,
            'columns': [],
            'sample_data': None,
            'data_types': {},
            'row_count': 0,
            'has_scores': False,
            'score_columns': [],
            'essay_columns': [],
            'prompt_columns': [],
            'feedback_columns': []
        }
        
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows for analysis
                analysis.update(self._analyze_csv_structure(df, file_path))
            elif file_path.suffix == '.jsonl':
                analysis.update(self._analyze_jsonl_structure(file_path))
            elif file_path.suffix == '.json':
                analysis.update(self._analyze_json_structure(file_path))
            elif file_path.suffix == '.xlsx':
                analysis.update(self._analyze_excel_structure(file_path))
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_csv_structure(self, df: pd.DataFrame, file_path: Path) -> Dict[str, Any]:
        """Analyze CSV file structure"""
        analysis = {}
        
        # Get full row count
        try:
            full_df = pd.read_csv(file_path)
            analysis['row_count'] = len(full_df)
        except:
            analysis['row_count'] = len(df)
        
        analysis['columns'] = list(df.columns)
        analysis['data_types'] = df.dtypes.to_dict()
        analysis['sample_data'] = df.head(2).to_dict('records')
        
        # Identify column types
        analysis['essay_columns'] = self._identify_essay_columns(df.columns)
        analysis['prompt_columns'] = self._identify_prompt_columns(df.columns)
        analysis['score_columns'] = self._identify_score_columns(df.columns)
        analysis['feedback_columns'] = self._identify_feedback_columns(df.columns)
        analysis['has_scores'] = len(analysis['score_columns']) > 0
        
        return analysis
    
    def _analyze_jsonl_structure(self, file_path: Path) -> Dict[str, Any]:
        """Analyze JSONL file structure"""
        analysis = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                analysis['row_count'] = len(lines)
                
                # Analyze first few lines
                sample_data = []
                for i, line in enumerate(lines[:3]):
                    try:
                        data = json.loads(line.strip())
                        sample_data.append(data)
                    except:
                        continue
                
                analysis['sample_data'] = sample_data
                
                if sample_data:
                    # Extract keys from sample data
                    all_keys = set()
                    for item in sample_data:
                        if isinstance(item, dict):
                            all_keys.update(item.keys())
                    
                    analysis['columns'] = list(all_keys)
                    
                    # Identify column types
                    analysis['essay_columns'] = self._identify_essay_columns(all_keys)
                    analysis['prompt_columns'] = self._identify_prompt_columns(all_keys)
                    analysis['score_columns'] = self._identify_score_columns(all_keys)
                    analysis['feedback_columns'] = self._identify_feedback_columns(all_keys)
                    analysis['has_scores'] = len(analysis['score_columns']) > 0
                
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_json_structure(self, file_path: Path) -> Dict[str, Any]:
        """Analyze JSON file structure"""
        analysis = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    analysis['row_count'] = len(data)
                    analysis['sample_data'] = data[:3]
                    
                    if data:
                        all_keys = set()
                        for item in data:
                            if isinstance(item, dict):
                                all_keys.update(item.keys())
                        
                        analysis['columns'] = list(all_keys)
                        analysis['essay_columns'] = self._identify_essay_columns(all_keys)
                        analysis['prompt_columns'] = self._identify_prompt_columns(all_keys)
                        analysis['score_columns'] = self._identify_score_columns(all_keys)
                        analysis['feedback_columns'] = self._identify_feedback_columns(all_keys)
                        analysis['has_scores'] = len(analysis['score_columns']) > 0
                else:
                    analysis['row_count'] = 1
                    analysis['sample_data'] = [data]
                    analysis['columns'] = list(data.keys()) if isinstance(data, dict) else []
        
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_excel_structure(self, file_path: Path) -> Dict[str, Any]:
        """Analyze Excel file structure"""
        analysis = {}
        
        try:
            # Read first sheet
            df = pd.read_excel(file_path, nrows=5)
            analysis['row_count'] = len(pd.read_excel(file_path))
            analysis['columns'] = list(df.columns)
            analysis['data_types'] = df.dtypes.to_dict()
            analysis['sample_data'] = df.head(2).to_dict('records')
            
            # Identify column types
            analysis['essay_columns'] = self._identify_essay_columns(df.columns)
            analysis['prompt_columns'] = self._identify_prompt_columns(df.columns)
            analysis['score_columns'] = self._identify_score_columns(df.columns)
            analysis['feedback_columns'] = self._identify_feedback_columns(df.columns)
            analysis['has_scores'] = len(analysis['score_columns']) > 0
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _identify_essay_columns(self, columns) -> List[str]:
        """Identify columns that likely contain essays"""
        essay_keywords = ['essay', 'text', 'response', 'answer', 'input', 'content', 'writing']
        return [col for col in columns if any(keyword in col.lower() for keyword in essay_keywords)]
    
    def _identify_prompt_columns(self, columns) -> List[str]:
        """Identify columns that likely contain prompts"""
        prompt_keywords = ['prompt', 'question', 'topic', 'task', 'instruction']
        return [col for col in columns if any(keyword in col.lower() for keyword in prompt_keywords)]
    
    def _identify_score_columns(self, columns) -> List[str]:
        """Identify columns that likely contain scores"""
        score_keywords = ['score', 'band', 'grade', 'rating', 'mark', 'achievement', 'coherence', 'lexical', 'grammatical', 'overall']
        return [col for col in columns if any(keyword in col.lower() for keyword in score_keywords)]
    
    def _identify_feedback_columns(self, columns) -> List[str]:
        """Identify columns that likely contain feedback"""
        feedback_keywords = ['feedback', 'comment', 'evaluation', 'output', 'review', 'assessment']
        return [col for col in columns if any(keyword in col.lower() for keyword in feedback_keywords)]
    
    def _print_file_summary(self, filename: str, analysis: Dict[str, Any]):
        """Print summary of file analysis"""
        if 'error' in analysis:
            print(f"❌ Error: {analysis['error']}")
            return
        
        print(f"📊 Rows: {analysis['row_count']:,}")
        print(f"📋 Columns: {len(analysis['columns'])}")
        print(f"📝 Essay columns: {analysis['essay_columns']}")
        print(f"❓ Prompt columns: {analysis['prompt_columns']}")
        print(f"📈 Score columns: {analysis['score_columns']}")
        print(f"💬 Feedback columns: {analysis['feedback_columns']}")
        print(f"🎯 Has scores: {'✅' if analysis['has_scores'] else '❌'}")
        
        if analysis['sample_data']:
            print("📄 Sample data preview:")
            for i, sample in enumerate(analysis['sample_data'][:1]):
                if isinstance(sample, dict):
                    for key, value in list(sample.items())[:3]:
                        preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        print(f"   {key}: {preview}")
    
    def generate_data_summary(self):
        """Generate overall data summary"""
        print("\n" + "=" * 60)
        print("📊 OVERALL DATA SUMMARY")
        print("=" * 60)
        
        total_files = len(self.file_analysis)
        files_with_scores = sum(1 for analysis in self.file_analysis.values() 
                              if analysis.get('has_scores', False))
        total_rows = sum(analysis.get('row_count', 0) for analysis in self.file_analysis.values())
        
        print(f"📁 Total files analyzed: {total_files}")
        print(f"📈 Files with scores: {files_with_scores}")
        print(f"📊 Total data rows: {total_rows:,}")
        
        # Find best files for training
        print("\n🎯 RECOMMENDED FILES FOR TRAINING:")
        print("-" * 40)
        
        training_candidates = []
        for filename, analysis in self.file_analysis.items():
            if analysis.get('has_scores', False) and analysis.get('row_count', 0) > 10:
                score = analysis.get('row_count', 0) * 0.7  # Weight by row count
                if len(analysis.get('essay_columns', [])) > 0:
                    score += 50  # Bonus for having essays
                if len(analysis.get('score_columns', [])) >= 4:
                    score += 100  # Bonus for having multiple score criteria
                
                training_candidates.append((filename, score, analysis))
        
        # Sort by score
        training_candidates.sort(key=lambda x: x[1], reverse=True)
        
        for i, (filename, score, analysis) in enumerate(training_candidates[:5]):
            print(f"{i+1}. {filename}")
            print(f"   📊 Rows: {analysis['row_count']:,}")
            print(f"   📝 Essays: {len(analysis['essay_columns'])} columns")
            print(f"   📈 Scores: {len(analysis['score_columns'])} columns")
            print(f"   🎯 Quality Score: {score:.1f}")
            print()
        
        return training_candidates

def main():
    """Main analysis function"""
    data_dir = "/Users/shan/Desktop/Work/Projects/EdPrep AI/IELTS/Writing/data_writing"
    
    analyzer = IELTSDataAnalyzer(data_dir)
    file_analysis = analyzer.analyze_all_files()
    training_candidates = analyzer.generate_data_summary()
    
    # Save analysis results
    output_dir = Path("/Users/shan/Desktop/Work/Projects/EdPrep AI/edprep-ai-prototype/data")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed analysis
    with open(output_dir / "data_analysis.json", "w") as f:
        json.dump(file_analysis, f, indent=2, default=str)
    
    # Save training recommendations
    recommendations = {
        "top_files": [
            {
                "filename": filename,
                "score": score,
                "row_count": analysis["row_count"],
                "essay_columns": analysis["essay_columns"],
                "score_columns": analysis["score_columns"]
            }
            for filename, score, analysis in training_candidates[:5]
        ]
    }
    
    with open(output_dir / "training_recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)
    
    print(f"\n💾 Analysis saved to:")
    print(f"   📄 {output_dir / 'data_analysis.json'}")
    print(f"   📄 {output_dir / 'training_recommendations.json'}")
    
    return file_analysis, training_candidates

if __name__ == "__main__":
    main()
