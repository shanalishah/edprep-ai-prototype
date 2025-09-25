#!/usr/bin/env python3
"""
Comprehensive Testing Framework: Official Rubric vs ML Models
Tests different weight combinations to find the most strict and authentic scoring approach
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our scoring models
from models.essay_scorer import EssayScorer
from models.ml_essay_scorer import MLEssayScorer
from models.production_essay_scorer import ProductionEssayScorer

class RubricMLTester:
    """
    Comprehensive testing framework to compare rubric-based vs ML-based scoring
    """
    
    def __init__(self):
        self.rubric_scorer = EssayScorer()
        self.ml_scorer = MLEssayScorer()
        self.production_scorer = ProductionEssayScorer()
        
        # Test essays with known quality levels
        self.test_essays = self._create_test_essays()
        
    def _create_test_essays(self) -> List[Dict[str, Any]]:
        """
        Create a comprehensive set of test essays with different quality levels
        """
        return [
            # HIGH QUALITY ESSAYS (Expected Band 7-9)
            {
                "id": "high_1",
                "prompt": "Some people think that the best way to reduce crime is to give longer prison sentences. Others, however, believe there are better alternative ways of reducing crime. Discuss both views and give your opinion.",
                "essay": """Crime is a pervasive issue that affects societies worldwide, and there is ongoing debate about the most effective methods to address it. While some advocate for longer prison sentences as a deterrent, others argue for alternative approaches that focus on rehabilitation and prevention.

Proponents of longer prison sentences argue that they serve as a strong deterrent to potential criminals. The fear of extended incarceration, they claim, discourages individuals from engaging in criminal activities. Additionally, longer sentences remove dangerous individuals from society for extended periods, providing immediate protection to the public. For example, countries with strict sentencing laws often report lower crime rates in certain categories.

However, critics of this approach point out several significant flaws. Research consistently shows that the threat of punishment alone is not an effective deterrent, particularly for crimes committed in the heat of the moment or by individuals with mental health issues. Moreover, longer sentences often lead to overcrowded prisons, which can become breeding grounds for further criminal behavior and radicalization.

Alternative approaches, such as community service, rehabilitation programs, and addressing root causes like poverty and lack of education, have shown promising results. Countries like Norway, which focus on rehabilitation rather than punishment, have significantly lower recidivism rates. These programs address the underlying factors that contribute to criminal behavior, creating lasting change rather than temporary containment.

In conclusion, while longer prison sentences may provide short-term protection, they fail to address the root causes of crime. A comprehensive approach that combines appropriate punishment with rehabilitation and prevention programs would be more effective in creating a safer society.""",
                "expected_band": 8.0,
                "quality": "high"
            },
            
            {
                "id": "high_2", 
                "prompt": "In many countries, more and more young people are leaving school and unable to find jobs after graduation. What problems do you think youth unemployment will cause to the individual and society? Give reasons and make suggestions.",
                "essay": """Youth unemployment has emerged as a critical challenge in contemporary society, with far-reaching implications for both individuals and communities. This phenomenon, driven by economic instability, technological disruption, and educational mismatches, requires urgent attention and comprehensive solutions.

For individuals, unemployment during the formative years can have devastating psychological and economic consequences. Young people who cannot secure employment often experience diminished self-esteem, anxiety, and depression, which can persist throughout their lives. Economically, they miss crucial opportunities to develop professional skills and build financial stability, creating a cycle of disadvantage that is difficult to break. Furthermore, prolonged unemployment can lead to social isolation and increased vulnerability to substance abuse or criminal activities.

Society faces equally significant challenges from youth unemployment. High unemployment rates among young people can lead to social unrest, as frustrated youth may express their discontent through protests or civil disobedience. Economically, society loses the productive potential of an entire generation, reducing overall economic growth and innovation. Additionally, governments must bear the cost of unemployment benefits and social services, while simultaneously losing tax revenue from unemployed individuals.

The root causes of youth unemployment are multifaceted. Educational systems often fail to align with labor market demands, producing graduates with skills that are not relevant to available positions. Economic downturns disproportionately affect entry-level positions, making it difficult for new graduates to find employment. Furthermore, rapid technological advancement has automated many traditional jobs while creating new opportunities that require different skill sets.

To address this crisis, governments must implement comprehensive strategies. Educational reform is essential to ensure that curricula prepare students for the evolving job market. Investment in vocational training and apprenticeship programs can provide practical skills and work experience. Additionally, policies that encourage entrepreneurship and support small businesses can create new employment opportunities. Finally, social safety nets must be strengthened to support young people during periods of unemployment while they develop new skills.

In conclusion, youth unemployment represents a complex challenge that requires coordinated efforts from governments, educational institutions, and the private sector. By addressing both the immediate needs of unemployed youth and the underlying structural issues, societies can unlock the potential of their young populations and build more resilient economies.""",
                "expected_band": 8.5,
                "quality": "high"
            },
            
            # MEDIUM QUALITY ESSAYS (Expected Band 5-6)
            {
                "id": "medium_1",
                "prompt": "Some people believe that technology has made our lives more complicated, while others think it has made our lives easier. Discuss both views and give your opinion.",
                "essay": """Technology is everywhere in our modern world. Some people think it makes life harder, but others believe it makes things easier. I think both views have some truth.

On one hand, technology can make life more complicated. People spend too much time on their phones and computers. They don't talk to each other face to face anymore. Also, technology changes very fast, so people have to learn new things all the time. This can be stressful and confusing.

On the other hand, technology makes many things easier. We can communicate with people around the world instantly. We can find information quickly on the internet. We can shop online without leaving our homes. Technology helps us work more efficiently and saves time.

However, there are also problems with technology. Sometimes it doesn't work properly and causes frustration. People become dependent on technology and can't do simple things without it. Also, technology can be expensive and not everyone can afford it.

In my opinion, technology is both good and bad. It makes some things easier but also creates new problems. The important thing is to use technology wisely and not let it control our lives. We should find a balance between using technology and living a simple life.

In conclusion, technology has both positive and negative effects on our lives. While it makes many things easier, it also creates new challenges. People should learn to use technology in a way that benefits them without causing problems.""",
                "expected_band": 6.0,
                "quality": "medium"
            },
            
            # LOW QUALITY ESSAYS (Expected Band 3-4)
            {
                "id": "low_1",
                "prompt": "Many people believe that social media has a negative impact on society. To what extent do you agree or disagree?",
                "essay": """Social media is bad for society. I agree with this statement because social media cause many problems.

First, social media make people waste time. People spend hours looking at their phones instead of doing important things. They don't study or work properly because they are always checking social media.

Second, social media make people feel bad about themselves. When people see other people's perfect photos, they feel sad because their life is not perfect. This can cause depression and anxiety.

Third, social media spread fake news. Many people believe everything they read on social media without checking if it is true. This can cause confusion and problems in society.

Also, social media make people less social. Instead of talking to friends in real life, people just send messages online. This is not good for relationships.

In conclusion, social media is very bad for society. It waste time, make people feel bad, spread fake news, and make people less social. People should use social media less and focus on real life instead.""",
                "expected_band": 4.0,
                "quality": "low"
            },
            
            {
                "id": "low_2",
                "prompt": "Some people think that governments should spend money on public transportation, while others believe that money should be spent on building more roads. Discuss both views and give your opinion.",
                "essay": """Government should spend money on roads not public transport. I think this because roads are more important.

Roads help everyone. Cars, buses, trucks all use roads. If roads are good, people can travel easily. Public transport only help some people who use buses and trains.

Building roads create jobs. Many people can work in construction. This help economy. Public transport also create jobs but not as many.

Roads are cheaper to build. You just need to put asphalt on ground. Public transport need expensive trains and stations. This cost too much money.

However, public transport is good for environment. Buses and trains don't pollute as much as cars. But most people still use cars anyway, so we need good roads.

In conclusion, government should spend money on roads because they help more people, create more jobs, and cost less money. Public transport is good but roads are more important for most people.""",
                "expected_band": 3.5,
                "quality": "low"
            }
        ]
    
    def test_single_essay(self, essay_data: Dict[str, Any], rubric_weight: float = 0.5) -> Dict[str, Any]:
        """
        Test a single essay with different scoring approaches and weight combinations
        """
        prompt = essay_data["prompt"]
        essay = essay_data["essay"]
        task_type = "Task 2"
        
        results = {
            "essay_id": essay_data["id"],
            "quality": essay_data["quality"],
            "expected_band": essay_data["expected_band"],
            "rubric_weight": rubric_weight,
            "ml_weight": 1.0 - rubric_weight
        }
        
        # Get rubric-based scores
        try:
            rubric_scores = self.rubric_scorer.score_essay(prompt, essay, task_type)
            results["rubric_scores"] = rubric_scores
        except Exception as e:
            results["rubric_scores"] = {"error": str(e)}
        
        # Get ML-based scores (try production first, then basic)
        ml_scores = {}
        if self.production_scorer.is_loaded:
            try:
                ml_scores = self.production_scorer.score_essay_production(essay, prompt, task_type)
                results["ml_method"] = "production"
            except Exception as e:
                ml_scores = {"error": str(e)}
        elif self.ml_scorer.is_loaded:
            try:
                ml_scores = self.ml_scorer.score_essay_ml(essay, prompt, task_type)
                results["ml_method"] = "basic"
            except Exception as e:
                ml_scores = {"error": str(e)}
        else:
            ml_scores = {"error": "No ML models loaded"}
            results["ml_method"] = "none"
        
        results["ml_scores"] = ml_scores
        
        # Calculate weighted combination if both methods work
        if "error" not in results["rubric_scores"] and "error" not in results["ml_scores"]:
            combined_scores = {}
            for criterion in ["task_achievement", "coherence_cohesion", "lexical_resource", "grammatical_range", "overall_band_score"]:
                if criterion in rubric_scores and criterion in ml_scores:
                    combined_scores[criterion] = (
                        rubric_weight * rubric_scores[criterion] + 
                        (1 - rubric_weight) * ml_scores[criterion]
                    )
            results["combined_scores"] = combined_scores
            
            # Calculate accuracy metrics
            expected = essay_data["expected_band"]
            rubric_error = abs(rubric_scores["overall_band_score"] - expected)
            ml_error = abs(ml_scores["overall_band_score"] - expected)
            combined_error = abs(combined_scores["overall_band_score"] - expected)
            
            results["accuracy_metrics"] = {
                "rubric_error": rubric_error,
                "ml_error": ml_error,
                "combined_error": combined_error,
                "rubric_strictness": self._calculate_strictness(rubric_scores["overall_band_score"], expected),
                "ml_strictness": self._calculate_strictness(ml_scores["overall_band_score"], expected),
                "combined_strictness": self._calculate_strictness(combined_scores["overall_band_score"], expected)
            }
        
        return results
    
    def _calculate_strictness(self, predicted: float, expected: float) -> str:
        """
        Calculate how strict the scoring is compared to expected
        """
        diff = predicted - expected
        if diff <= -0.5:
            return "very_strict"
        elif diff <= -0.25:
            return "strict"
        elif diff <= 0.25:
            return "accurate"
        elif diff <= 0.5:
            return "lenient"
        else:
            return "very_lenient"
    
    def test_weight_combinations(self, weights: List[float] = None) -> pd.DataFrame:
        """
        Test different weight combinations between rubric and ML approaches
        """
        if weights is None:
            weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        all_results = []
        
        print("🧪 Testing Weight Combinations...")
        print("=" * 60)
        
        for weight in weights:
            print(f"\n📊 Testing Rubric Weight: {weight:.1f}, ML Weight: {1-weight:.1f}")
            
            for essay in self.test_essays:
                result = self.test_single_essay(essay, weight)
                all_results.append(result)
                
                # Print summary for this essay
                if "accuracy_metrics" in result:
                    metrics = result["accuracy_metrics"]
                    print(f"  {essay['id']} ({essay['quality']}): "
                          f"Expected={essay['expected_band']:.1f}, "
                          f"Rubric={result['rubric_scores']['overall_band_score']:.1f}, "
                          f"ML={result['ml_scores']['overall_band_score']:.1f}, "
                          f"Combined={result['combined_scores']['overall_band_score']:.1f}")
        
        return pd.DataFrame(all_results)
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the test results to find optimal weights
        """
        print("\n📈 ANALYZING RESULTS...")
        print("=" * 60)
        
        # Filter out failed tests
        valid_results = results_df[
            results_df['accuracy_metrics'].notna() & 
            results_df['rubric_scores'].apply(lambda x: 'error' not in x if isinstance(x, dict) else False) &
            results_df['ml_scores'].apply(lambda x: 'error' not in x if isinstance(x, dict) else False)
        ].copy()
        
        if valid_results.empty:
            return {"error": "No valid results to analyze"}
        
        # Extract accuracy metrics
        valid_results['rubric_error'] = valid_results['accuracy_metrics'].apply(lambda x: x['rubric_error'])
        valid_results['ml_error'] = valid_results['accuracy_metrics'].apply(lambda x: x['ml_error'])
        valid_results['combined_error'] = valid_results['accuracy_metrics'].apply(lambda x: x['combined_error'])
        valid_results['rubric_strictness'] = valid_results['accuracy_metrics'].apply(lambda x: x['rubric_strictness'])
        valid_results['ml_strictness'] = valid_results['accuracy_metrics'].apply(lambda x: x['ml_strictness'])
        valid_results['combined_strictness'] = valid_results['accuracy_metrics'].apply(lambda x: x['combined_strictness'])
        
        # Calculate average errors by weight
        weight_analysis = valid_results.groupby('rubric_weight').agg({
            'rubric_error': 'mean',
            'ml_error': 'mean', 
            'combined_error': 'mean',
            'rubric_strictness': lambda x: (x == 'strict').sum() / len(x),
            'ml_strictness': lambda x: (x == 'strict').sum() / len(x),
            'combined_strictness': lambda x: (x == 'strict').sum() / len(x)
        }).round(3)
        
        print("\n📊 WEIGHT ANALYSIS:")
        print(weight_analysis)
        
        # Find optimal weights
        best_accuracy_weight = weight_analysis['combined_error'].idxmin()
        best_strictness_weight = weight_analysis['combined_strictness'].idxmax()
        
        # Analyze by essay quality
        quality_analysis = valid_results.groupby(['quality', 'rubric_weight']).agg({
            'combined_error': 'mean',
            'combined_strictness': lambda x: (x == 'strict').sum() / len(x)
        }).round(3)
        
        print(f"\n🎯 OPTIMAL WEIGHTS:")
        print(f"Best Accuracy: Rubric Weight = {best_accuracy_weight:.1f}")
        print(f"Best Strictness: Rubric Weight = {best_strictness_weight:.1f}")
        
        # Detailed analysis by quality level
        print(f"\n📋 ANALYSIS BY QUALITY LEVEL:")
        for quality in ['high', 'medium', 'low']:
            quality_data = valid_results[valid_results['quality'] == quality]
            if not quality_data.empty:
                best_weight = quality_data.groupby('rubric_weight')['combined_error'].mean().idxmin()
                avg_error = quality_data.groupby('rubric_weight')['combined_error'].mean().min()
                print(f"{quality.upper()} essays: Best weight = {best_weight:.1f}, Avg error = {avg_error:.3f}")
        
        return {
            "weight_analysis": weight_analysis,
            "best_accuracy_weight": best_accuracy_weight,
            "best_strictness_weight": best_strictness_weight,
            "quality_analysis": quality_analysis,
            "valid_results": valid_results
        }
    
    def create_visualizations(self, results_df: pd.DataFrame, analysis: Dict[str, Any]):
        """
        Create visualizations of the test results
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Rubric vs ML Scoring Analysis', fontsize=16, fontweight='bold')
            
            # Filter valid results
            valid_results = results_df[
                results_df['accuracy_metrics'].notna() & 
                results_df['rubric_scores'].apply(lambda x: 'error' not in x if isinstance(x, dict) else False) &
                results_df['ml_scores'].apply(lambda x: 'error' not in x if isinstance(x, dict) else False)
            ].copy()
            
            if valid_results.empty:
                print("No valid results for visualization")
                return
            
            # Extract metrics
            valid_results['rubric_error'] = valid_results['accuracy_metrics'].apply(lambda x: x['rubric_error'])
            valid_results['ml_error'] = valid_results['accuracy_metrics'].apply(lambda x: x['ml_error'])
            valid_results['combined_error'] = valid_results['accuracy_metrics'].apply(lambda x: x['combined_error'])
            
            # Plot 1: Error by Weight
            weight_errors = valid_results.groupby('rubric_weight')[['rubric_error', 'ml_error', 'combined_error']].mean()
            weight_errors.plot(ax=axes[0,0], marker='o', linewidth=2)
            axes[0,0].set_title('Average Error by Rubric Weight')
            axes[0,0].set_xlabel('Rubric Weight')
            axes[0,0].set_ylabel('Average Error')
            axes[0,0].legend(['Rubric Only', 'ML Only', 'Combined'])
            axes[0,0].grid(True, alpha=0.3)
            
            # Plot 2: Error by Essay Quality
            quality_errors = valid_results.groupby(['quality', 'rubric_weight'])['combined_error'].mean().unstack()
            quality_errors.plot(ax=axes[0,1], marker='o', linewidth=2)
            axes[0,1].set_title('Error by Essay Quality and Weight')
            axes[0,1].set_xlabel('Rubric Weight')
            axes[0,1].set_ylabel('Average Error')
            axes[0,1].legend(['High Quality', 'Medium Quality', 'Low Quality'])
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3: Score Distribution
            essay_scores = []
            for _, row in valid_results.iterrows():
                essay_scores.append({
                    'Essay': row['essay_id'],
                    'Quality': row['quality'],
                    'Expected': row['expected_band'],
                    'Rubric': row['rubric_scores']['overall_band_score'],
                    'ML': row['ml_scores']['overall_band_score'],
                    'Combined': row['combined_scores']['overall_band_score']
                })
            
            scores_df = pd.DataFrame(essay_scores)
            scores_melted = scores_df.melt(id_vars=['Essay', 'Quality', 'Expected'], 
                                         value_vars=['Rubric', 'ML', 'Combined'],
                                         var_name='Method', value_name='Score')
            
            sns.boxplot(data=scores_melted, x='Method', y='Score', hue='Quality', ax=axes[1,0])
            axes[1,0].set_title('Score Distribution by Method and Quality')
            axes[1,0].set_ylabel('Band Score')
            
            # Plot 4: Strictness Analysis
            strictness_counts = valid_results.groupby(['rubric_weight', 'combined_strictness']).size().unstack(fill_value=0)
            strictness_counts.plot(kind='bar', stacked=True, ax=axes[1,1])
            axes[1,1].set_title('Strictness Distribution by Weight')
            axes[1,1].set_xlabel('Rubric Weight')
            axes[1,1].set_ylabel('Number of Essays')
            axes[1,1].legend(title='Strictness Level')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save the plot
            output_path = Path(__file__).parent / 'rubric_vs_ml_analysis.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Visualization saved to: {output_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib/Seaborn not available for visualization")
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    def run_comprehensive_test(self):
        """
        Run the complete testing framework
        """
        print("🚀 STARTING COMPREHENSIVE RUBRIC vs ML TESTING")
        print("=" * 80)
        
        # Test different weight combinations
        weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        results_df = self.test_weight_combinations(weights)
        
        # Analyze results
        analysis = self.analyze_results(results_df)
        
        # Create visualizations
        self.create_visualizations(results_df, analysis)
        
        # Save results
        output_path = Path(__file__).parent / 'rubric_vs_ml_results.json'
        results_df.to_json(output_path, orient='records', indent=2)
        print(f"\n💾 Results saved to: {output_path}")
        
        return results_df, analysis

def main():
    """
    Main function to run the testing framework
    """
    print("🎯 EdPrep AI: Rubric vs ML Scoring Analysis")
    print("=" * 50)
    
    # Initialize tester
    tester = RubricMLTester()
    
    # Run comprehensive test
    results_df, analysis = tester.run_comprehensive_test()
    
    print("\n✅ Testing Complete!")
    print("Check the generated files for detailed results and visualizations.")

if __name__ == "__main__":
    main()
