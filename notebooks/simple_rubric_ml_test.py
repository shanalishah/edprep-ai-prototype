#!/usr/bin/env python3
"""
Simplified Testing: Official Rubric vs ML Models
Focus on core comparison without complex feature engineering issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Import our scoring models
from models.essay_scorer import EssayScorer
from models.ml_essay_scorer import MLEssayScorer

class SimpleRubricMLTester:
    """
    Simplified testing framework focusing on core rubric vs ML comparison
    """
    
    def __init__(self):
        self.rubric_scorer = EssayScorer()
        self.ml_scorer = MLEssayScorer()
        
        # Test essays with known quality levels
        self.test_essays = self._create_test_essays()
        
    def _create_test_essays(self) -> List[Dict[str, Any]]:
        """
        Create a focused set of test essays with different quality levels
        """
        return [
            # HIGH QUALITY ESSAY (Expected Band 8-9)
            {
                "id": "high_quality",
                "prompt": "Some people think that the best way to reduce crime is to give longer prison sentences. Others, however, believe there are better alternative ways of reducing crime. Discuss both views and give your opinion.",
                "essay": """Crime is a pervasive issue that affects societies worldwide, and there is ongoing debate about the most effective methods to address it. While some advocate for longer prison sentences as a deterrent, others argue for alternative approaches that focus on rehabilitation and prevention.

Proponents of longer prison sentences argue that they serve as a strong deterrent to potential criminals. The fear of extended incarceration, they claim, discourages individuals from engaging in criminal activities. Additionally, longer sentences remove dangerous individuals from society for extended periods, providing immediate protection to the public. For example, countries with strict sentencing laws often report lower crime rates in certain categories.

However, critics of this approach point out several significant flaws. Research consistently shows that the threat of punishment alone is not an effective deterrent, particularly for crimes committed in the heat of the moment or by individuals with mental health issues. Moreover, longer sentences often lead to overcrowded prisons, which can become breeding grounds for further criminal behavior and radicalization.

Alternative approaches, such as community service, rehabilitation programs, and addressing root causes like poverty and lack of education, have shown promising results. Countries like Norway, which focus on rehabilitation rather than punishment, have significantly lower recidivism rates. These programs address the underlying factors that contribute to criminal behavior, creating lasting change rather than temporary containment.

In conclusion, while longer prison sentences may provide short-term protection, they fail to address the root causes of crime. A comprehensive approach that combines appropriate punishment with rehabilitation and prevention programs would be more effective in creating a safer society.""",
                "expected_band": 8.0,
                "quality": "high",
                "word_count": 280
            },
            
            # MEDIUM QUALITY ESSAY (Expected Band 5-6)
            {
                "id": "medium_quality",
                "prompt": "Some people believe that technology has made our lives more complicated, while others think it has made our lives easier. Discuss both views and give your opinion.",
                "essay": """Technology is everywhere in our modern world. Some people think it makes life harder, but others believe it makes things easier. I think both views have some truth.

On one hand, technology can make life more complicated. People spend too much time on their phones and computers. They don't talk to each other face to face anymore. Also, technology changes very fast, so people have to learn new things all the time. This can be stressful and confusing.

On the other hand, technology makes many things easier. We can communicate with people around the world instantly. We can find information quickly on the internet. We can shop online without leaving our homes. Technology helps us work more efficiently and saves time.

However, there are also problems with technology. Sometimes it doesn't work properly and causes frustration. People become dependent on technology and can't do simple things without it. Also, technology can be expensive and not everyone can afford it.

In my opinion, technology is both good and bad. It makes some things easier but also creates new problems. The important thing is to use technology wisely and not let it control our lives. We should find a balance between using technology and living a simple life.

In conclusion, technology has both positive and negative effects on our lives. While it makes many things easier, it also creates new challenges. People should learn to use technology in a way that benefits them without causing problems.""",
                "expected_band": 6.0,
                "quality": "medium",
                "word_count": 220
            },
            
            # LOW QUALITY ESSAY (Expected Band 3-4)
            {
                "id": "low_quality",
                "prompt": "Many people believe that social media has a negative impact on society. To what extent do you agree or disagree?",
                "essay": """Social media is bad for society. I agree with this statement because social media cause many problems.

First, social media make people waste time. People spend hours looking at their phones instead of doing important things. They don't study or work properly because they are always checking social media.

Second, social media make people feel bad about themselves. When people see other people's perfect photos, they feel sad because their life is not perfect. This can cause depression and anxiety.

Third, social media spread fake news. Many people believe everything they read on social media without checking if it is true. This can cause confusion and problems in society.

Also, social media make people less social. Instead of talking to friends in real life, people just send messages online. This is not good for relationships.

In conclusion, social media is very bad for society. It waste time, make people feel bad, spread fake news, and make people less social. People should use social media less and focus on real life instead.""",
                "expected_band": 4.0,
                "quality": "low",
                "word_count": 150
            }
        ]
    
    def test_single_essay(self, essay_data: Dict[str, Any], rubric_weight: float = 0.5) -> Dict[str, Any]:
        """
        Test a single essay with different scoring approaches
        """
        prompt = essay_data["prompt"]
        essay = essay_data["essay"]
        task_type = "Task 2"
        
        results = {
            "essay_id": essay_data["id"],
            "quality": essay_data["quality"],
            "expected_band": essay_data["expected_band"],
            "word_count": essay_data["word_count"],
            "rubric_weight": rubric_weight,
            "ml_weight": 1.0 - rubric_weight
        }
        
        # Get rubric-based scores
        try:
            rubric_scores = self.rubric_scorer.score_essay(prompt, essay, task_type)
            results["rubric_scores"] = rubric_scores
        except Exception as e:
            results["rubric_scores"] = {"error": str(e)}
        
        # Get ML-based scores
        ml_scores = {}
        if self.ml_scorer.is_loaded:
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
    
    def run_comprehensive_test(self):
        """
        Run the complete testing framework
        """
        print("🚀 SIMPLIFIED RUBRIC vs ML TESTING")
        print("=" * 60)
        
        # Test different weight combinations
        weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        
        print("\n📊 DETAILED RESULTS BY ESSAY:")
        print("=" * 60)
        
        all_results = []
        
        for weight in weights:
            print(f"\n🔍 Testing Rubric Weight: {weight:.1f}, ML Weight: {1-weight:.1f}")
            print("-" * 50)
            
            for essay in self.test_essays:
                result = self.test_single_essay(essay, weight)
                all_results.append(result)
                
                # Print detailed results
                if "accuracy_metrics" in result:
                    metrics = result["accuracy_metrics"]
                    print(f"\n📝 {essay['id']} ({essay['quality']} quality, {essay['word_count']} words)")
                    print(f"   Expected Band: {essay['expected_band']:.1f}")
                    print(f"   Rubric Score: {result['rubric_scores']['overall_band_score']:.1f} (Error: {metrics['rubric_error']:.1f})")
                    print(f"   ML Score: {result['ml_scores']['overall_band_score']:.1f} (Error: {metrics['ml_error']:.1f})")
                    print(f"   Combined Score: {result['combined_scores']['overall_band_score']:.1f} (Error: {metrics['combined_error']:.1f})")
                    print(f"   Strictness: Rubric={metrics['rubric_strictness']}, ML={metrics['ml_strictness']}, Combined={metrics['combined_strictness']}")
                else:
                    print(f"\n❌ {essay['id']}: Scoring failed")
        
        # Analyze results
        print("\n📈 ANALYSIS SUMMARY:")
        print("=" * 60)
        
        # Filter valid results
        valid_results = [r for r in all_results if "accuracy_metrics" in r]
        
        if not valid_results:
            print("❌ No valid results to analyze")
            return
        
        # Calculate average errors by weight
        weight_analysis = {}
        for weight in weights:
            weight_results = [r for r in valid_results if r["rubric_weight"] == weight]
            if weight_results:
                avg_rubric_error = np.mean([r["accuracy_metrics"]["rubric_error"] for r in weight_results])
                avg_ml_error = np.mean([r["accuracy_metrics"]["ml_error"] for r in weight_results])
                avg_combined_error = np.mean([r["accuracy_metrics"]["combined_error"] for r in weight_results])
                
                strict_count = sum(1 for r in weight_results if r["accuracy_metrics"]["combined_strictness"] in ["strict", "very_strict"])
                strictness_ratio = strict_count / len(weight_results)
                
                weight_analysis[weight] = {
                    "avg_rubric_error": avg_rubric_error,
                    "avg_ml_error": avg_ml_error,
                    "avg_combined_error": avg_combined_error,
                    "strictness_ratio": strictness_ratio
                }
        
        # Print analysis
        print("\n📊 WEIGHT ANALYSIS:")
        print("Weight | Rubric Error | ML Error | Combined Error | Strictness")
        print("-" * 65)
        for weight in sorted(weight_analysis.keys()):
            analysis = weight_analysis[weight]
            print(f"{weight:6.1f} | {analysis['avg_rubric_error']:11.3f} | {analysis['avg_ml_error']:8.3f} | {analysis['avg_combined_error']:13.3f} | {analysis['strictness_ratio']:9.3f}")
        
        # Find optimal weights
        best_accuracy_weight = min(weight_analysis.keys(), key=lambda w: weight_analysis[w]["avg_combined_error"])
        best_strictness_weight = max(weight_analysis.keys(), key=lambda w: weight_analysis[w]["strictness_ratio"])
        
        print(f"\n🎯 OPTIMAL WEIGHTS:")
        print(f"Best Accuracy: Rubric Weight = {best_accuracy_weight:.1f} (Error: {weight_analysis[best_accuracy_weight]['avg_combined_error']:.3f})")
        print(f"Best Strictness: Rubric Weight = {best_strictness_weight:.1f} (Strictness: {weight_analysis[best_strictness_weight]['strictness_ratio']:.3f})")
        
        # Analysis by quality level
        print(f"\n📋 ANALYSIS BY QUALITY LEVEL:")
        for quality in ['high', 'medium', 'low']:
            quality_results = [r for r in valid_results if r["quality"] == quality]
            if quality_results:
                best_weight = min(quality_results, key=lambda r: r["accuracy_metrics"]["combined_error"])["rubric_weight"]
                avg_error = np.mean([r["accuracy_metrics"]["combined_error"] for r in quality_results])
                print(f"{quality.upper()} essays: Best weight = {best_weight:.1f}, Avg error = {avg_error:.3f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("=" * 60)
        
        if weight_analysis[best_accuracy_weight]["avg_combined_error"] < weight_analysis[1.0]["avg_rubric_error"]:
            print("✅ HYBRID APPROACH is more accurate than rubric-only")
        else:
            print("⚠️ RUBRIC-ONLY approach is more accurate")
        
        if weight_analysis[best_strictness_weight]["strictness_ratio"] > 0.5:
            print("✅ HYBRID APPROACH can provide stricter scoring")
        else:
            print("⚠️ HYBRID APPROACH tends to be lenient")
        
        print(f"\n🎯 RECOMMENDED CONFIGURATION:")
        print(f"For Accuracy: Use {best_accuracy_weight:.1f} rubric weight")
        print(f"For Strictness: Use {best_strictness_weight:.1f} rubric weight")
        print(f"For Balanced: Use 0.5-0.6 rubric weight")
        
        return all_results, weight_analysis

def main():
    """
    Main function to run the simplified testing
    """
    print("🎯 EdPrep AI: Simplified Rubric vs ML Analysis")
    print("=" * 50)
    
    # Initialize tester
    tester = SimpleRubricMLTester()
    
    # Run test
    results, analysis = tester.run_comprehensive_test()
    
    print("\n✅ Testing Complete!")
    print("This analysis helps determine the optimal balance between rubric-based and ML-based scoring.")

if __name__ == "__main__":
    main()
