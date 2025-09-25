#!/usr/bin/env python3
"""
API-Based Testing: Test the actual running system with different approaches
"""

import requests
import json
import time
from typing import Dict, List, Any

class APIBasedTester:
    """
    Test the actual running EdPrep AI API with different scoring approaches
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_essays = self._create_test_essays()
        
    def _create_test_essays(self) -> List[Dict[str, Any]]:
        """
        Create test essays with different quality levels
        """
        return [
            # HIGH QUALITY ESSAY
            {
                "id": "high_quality",
                "prompt": "Some people think that the best way to reduce crime is to give longer prison sentences. Others, however, believe there are better alternative ways of reducing crime. Discuss both views and give your opinion.",
                "essay": """Crime is a pervasive issue that affects societies worldwide, and there is ongoing debate about the most effective methods to address it. While some advocate for longer prison sentences as a deterrent, others argue for alternative approaches that focus on rehabilitation and prevention.

Proponents of longer prison sentences argue that they serve as a strong deterrent to potential criminals. The fear of extended incarceration, they claim, discourages individuals from engaging in criminal activities. Additionally, longer sentences remove dangerous individuals from society for extended periods, providing immediate protection to the public. For example, countries with strict sentencing laws often report lower crime rates in certain categories.

However, critics of this approach point out several significant flaws. Research consistently shows that the threat of punishment alone is not an effective deterrent, particularly for crimes committed in the heat of the moment or by individuals with mental health issues. Moreover, longer sentences often lead to overcrowded prisons, which can become breeding grounds for further criminal behavior and radicalization.

Alternative approaches, such as community service, rehabilitation programs, and addressing root causes like poverty and lack of education, have shown promising results. Countries like Norway, which focus on rehabilitation rather than punishment, have significantly lower recidivism rates. These programs address the underlying factors that contribute to criminal behavior, creating lasting change rather than temporary containment.

In conclusion, while longer prison sentences may provide short-term protection, they fail to address the root causes of crime. A comprehensive approach that combines appropriate punishment with rehabilitation and prevention programs would be more effective in creating a safer society.""",
                "expected_band": 8.0,
                "quality": "high"
            },
            
            # MEDIUM QUALITY ESSAY
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
                "quality": "medium"
            },
            
            # LOW QUALITY ESSAY
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
                "quality": "low"
            }
        ]
    
    def test_essay_assessment(self, essay_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test essay assessment using the main API endpoint
        """
        try:
            response = requests.post(
                f"{self.base_url}/assess",
                json={
                    "prompt": essay_data["prompt"],
                    "essay": essay_data["essay"],
                    "task_type": "Task 2"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "scores": {
                        "task_achievement": result["task_achievement"],
                        "coherence_cohesion": result["coherence_cohesion"],
                        "lexical_resource": result["lexical_resource"],
                        "grammatical_range": result["grammatical_range"],
                        "overall_band_score": result["overall_band_score"]
                    },
                    "feedback": result["feedback"],
                    "suggestions": result["suggestions"]
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_hybrid_weights(self, essay_data: Dict[str, Any], rubric_weight: float) -> Dict[str, Any]:
        """
        Test hybrid scoring with specific rubric weight
        """
        try:
            response = requests.post(
                f"{self.base_url}/test-hybrid-weights",
                params={"rubric_weight": rubric_weight},
                json={
                    "prompt": essay_data["prompt"],
                    "essay": essay_data["essay"],
                    "task_type": "Task 2"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "scores": result["scores"],
                    "scoring_details": result["scoring_details"],
                    "individual_scores": result["individual_scores"]
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current model status
        """
        try:
            response = requests.get(f"{self.base_url}/model-status", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def run_comprehensive_test(self):
        """
        Run comprehensive testing of the API
        """
        print("🚀 API-BASED RUBRIC vs ML TESTING")
        print("=" * 60)
        
        # Check if API is running
        print("🔍 Checking API status...")
        status = self.get_model_status()
        if "error" in status:
            print(f"❌ API not accessible: {status['error']}")
            print("Make sure the backend is running on http://localhost:8000")
            return
        
        print(f"✅ API is running")
        print(f"📊 Current scoring method: {status.get('scoring_method', 'Unknown')}")
        print(f"🔧 Features: {status.get('features', 'Unknown')}")
        
        # Test different weight combinations
        weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        
        print(f"\n📊 TESTING DIFFERENT WEIGHT COMBINATIONS:")
        print("=" * 60)
        
        all_results = []
        
        for weight in weights:
            print(f"\n🔍 Testing Rubric Weight: {weight:.1f}, ML Weight: {1-weight:.1f}")
            print("-" * 50)
            
            for essay in self.test_essays:
                print(f"\n📝 Testing {essay['id']} ({essay['quality']} quality)")
                
                # Test with hybrid weights
                result = self.test_hybrid_weights(essay, weight)
                
                if result["success"]:
                    scores = result["scores"]
                    overall_score = scores["overall_band_score"]
                    expected = essay["expected_band"]
                    error = abs(overall_score - expected)
                    
                    print(f"   Expected: {expected:.1f}, Got: {overall_score:.1f}, Error: {error:.1f}")
                    
                    # Calculate strictness
                    if error <= 0.25:
                        strictness = "accurate"
                    elif overall_score < expected - 0.25:
                        strictness = "strict"
                    else:
                        strictness = "lenient"
                    
                    print(f"   Strictness: {strictness}")
                    
                    all_results.append({
                        "essay_id": essay["id"],
                        "quality": essay["quality"],
                        "expected_band": expected,
                        "rubric_weight": weight,
                        "overall_score": overall_score,
                        "error": error,
                        "strictness": strictness,
                        "scores": scores
                    })
                else:
                    print(f"   ❌ Failed: {result['error']}")
        
        # Analyze results
        if all_results:
            print(f"\n📈 ANALYSIS SUMMARY:")
            print("=" * 60)
            
            # Calculate average errors by weight
            weight_analysis = {}
            for weight in weights:
                weight_results = [r for r in all_results if r["rubric_weight"] == weight]
                if weight_results:
                    avg_error = sum(r["error"] for r in weight_results) / len(weight_results)
                    strict_count = sum(1 for r in weight_results if r["strictness"] in ["strict", "accurate"])
                    strictness_ratio = strict_count / len(weight_results)
                    
                    weight_analysis[weight] = {
                        "avg_error": avg_error,
                        "strictness_ratio": strictness_ratio,
                        "count": len(weight_results)
                    }
            
            # Print analysis
            print("\n📊 WEIGHT ANALYSIS:")
            print("Weight | Avg Error | Strictness | Count")
            print("-" * 40)
            for weight in sorted(weight_analysis.keys()):
                analysis = weight_analysis[weight]
                print(f"{weight:6.1f} | {analysis['avg_error']:9.3f} | {analysis['strictness_ratio']:10.3f} | {analysis['count']:5d}")
            
            # Find optimal weights
            best_accuracy_weight = min(weight_analysis.keys(), key=lambda w: weight_analysis[w]["avg_error"])
            best_strictness_weight = max(weight_analysis.keys(), key=lambda w: weight_analysis[w]["strictness_ratio"])
            
            print(f"\n🎯 OPTIMAL WEIGHTS:")
            print(f"Best Accuracy: Rubric Weight = {best_accuracy_weight:.1f} (Error: {weight_analysis[best_accuracy_weight]['avg_error']:.3f})")
            print(f"Best Strictness: Rubric Weight = {best_strictness_weight:.1f} (Strictness: {weight_analysis[best_strictness_weight]['strictness_ratio']:.3f})")
            
            # Analysis by quality level
            print(f"\n📋 ANALYSIS BY QUALITY LEVEL:")
            for quality in ['high', 'medium', 'low']:
                quality_results = [r for r in all_results if r["quality"] == quality]
                if quality_results:
                    best_weight = min(quality_results, key=lambda r: r["error"])["rubric_weight"]
                    avg_error = sum(r["error"] for r in quality_results) / len(quality_results)
                    print(f"{quality.upper()} essays: Best weight = {best_weight:.1f}, Avg error = {avg_error:.3f}")
            
            # Recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            print("=" * 60)
            
            if weight_analysis[best_accuracy_weight]["avg_error"] < 1.0:
                print("✅ HYBRID APPROACH shows good accuracy")
            else:
                print("⚠️ HYBRID APPROACH needs improvement")
            
            if weight_analysis[best_strictness_weight]["strictness_ratio"] > 0.5:
                print("✅ HYBRID APPROACH can provide strict scoring")
            else:
                print("⚠️ HYBRID APPROACH tends to be lenient")
            
            print(f"\n🎯 RECOMMENDED CONFIGURATION:")
            print(f"For Accuracy: Use {best_accuracy_weight:.1f} rubric weight")
            print(f"For Strictness: Use {best_strictness_weight:.1f} rubric weight")
            print(f"For Balanced: Use 0.5-0.6 rubric weight")
            
            # Save results
            with open("api_test_results.json", "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\n💾 Results saved to api_test_results.json")
        
        else:
            print("❌ No successful test results to analyze")

def main():
    """
    Main function to run the API-based testing
    """
    print("🎯 EdPrep AI: API-Based Rubric vs ML Analysis")
    print("=" * 50)
    
    # Initialize tester
    tester = APIBasedTester()
    
    # Run test
    tester.run_comprehensive_test()
    
    print("\n✅ Testing Complete!")
    print("This analysis shows the actual performance of your running system.")

if __name__ == "__main__":
    main()
