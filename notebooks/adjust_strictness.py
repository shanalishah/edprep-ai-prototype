#!/usr/bin/env python3
"""
Adjust System Strictness for Real IELTS-like Assessment
"""

import requests
import json

def test_current_strictness():
    """
    Test current system strictness with different quality essays
    """
    test_essays = [
        {
            'id': 'high_quality',
            'prompt': 'Some people think that the best way to reduce crime is to give longer prison sentences. Others, however, believe there are better alternative ways of reducing crime. Discuss both views and give your opinion.',
            'essay': 'Crime is a pervasive issue that affects societies worldwide, and there is ongoing debate about the most effective methods to address it. While some advocate for longer prison sentences as a deterrent, others argue for alternative approaches that focus on rehabilitation and prevention. Proponents of longer prison sentences argue that they serve as a strong deterrent to potential criminals. The fear of extended incarceration, they claim, discourages individuals from engaging in criminal activities. Additionally, longer sentences remove dangerous individuals from society for extended periods, providing immediate protection to the public. For example, countries with strict sentencing laws often report lower crime rates in certain categories. However, critics of this approach point out several significant flaws. Research consistently shows that the threat of punishment alone is not an effective deterrent, particularly for crimes committed in the heat of the moment or by individuals with mental health issues. Moreover, longer sentences often lead to overcrowded prisons, which can become breeding grounds for further criminal behavior and radicalization. Alternative approaches, such as community service, rehabilitation programs, and addressing root causes like poverty and lack of education, have shown promising results. Countries like Norway, which focus on rehabilitation rather than punishment, have significantly lower recidivism rates. These programs address the underlying factors that contribute to criminal behavior, creating lasting change rather than temporary containment. In conclusion, while longer prison sentences may provide short-term protection, they fail to address the root causes of crime. A comprehensive approach that combines appropriate punishment with rehabilitation and prevention programs would be more effective in creating a safer society.',
            'expected_band': 8.0,
            'quality': 'high'
        },
        {
            'id': 'low_quality',
            'prompt': 'Many people believe that social media has a negative impact on society. To what extent do you agree or disagree?',
            'essay': 'Social media is bad for society. I agree with this statement because social media cause many problems. First, social media make people waste time. People spend hours looking at their phones instead of doing important things. They don\'t study or work properly because they are always checking social media. Second, social media make people feel bad about themselves. When people see other people\'s perfect photos, they feel sad because their life is not perfect. This can cause depression and anxiety. Third, social media spread fake news. Many people believe everything they read on social media without checking if it is true. This can cause confusion and problems in society. Also, social media make people less social. Instead of talking to friends in real life, people just send messages online. This is not good for relationships. In conclusion, social media is very bad for society. It waste time, make people feel bad, spread fake news, and make people less social. People should use social media less and focus on real life instead.',
            'expected_band': 4.0,
            'quality': 'low'
        }
    ]
    
    print("🧪 Testing Current System Strictness...")
    print("=" * 60)
    
    results = []
    
    for essay in test_essays:
        print(f"\n📝 Testing {essay['id']} ({essay['quality']} quality)")
        print(f"Expected Band: {essay['expected_band']:.1f}")
        
        try:
            response = requests.post('http://localhost:8000/assess', json={
                'prompt': essay['prompt'],
                'essay': essay['essay'],
                'task_type': 'Task 2'
            }, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                overall_score = result['overall_band_score']
                error = abs(overall_score - essay['expected_band'])
                
                print(f"✅ Got Band Score: {overall_score:.1f}")
                print(f"📊 Error: {error:.1f}")
                
                # Calculate strictness
                if error <= 0.25:
                    strictness = 'accurate'
                elif overall_score < essay['expected_band'] - 0.25:
                    strictness = 'strict'
                else:
                    strictness = 'lenient'
                
                print(f"🎯 Strictness: {strictness}")
                
                results.append({
                    'essay_id': essay['id'],
                    'quality': essay['quality'],
                    'expected': essay['expected_band'],
                    'actual': overall_score,
                    'error': error,
                    'strictness': strictness
                })
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return results

def analyze_strictness(results):
    """
    Analyze the strictness results and provide recommendations
    """
    print(f"\n📈 STRICTNESS ANALYSIS:")
    print("=" * 60)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Calculate overall strictness
    strict_count = sum(1 for r in results if r['strictness'] == 'strict')
    lenient_count = sum(1 for r in results if r['strictness'] == 'lenient')
    accurate_count = sum(1 for r in results if r['strictness'] == 'accurate')
    
    total = len(results)
    
    print(f"📊 Strictness Distribution:")
    print(f"  Strict: {strict_count}/{total} ({strict_count/total*100:.1f}%)")
    print(f"  Lenient: {lenient_count}/{total} ({lenient_count/total*100:.1f}%)")
    print(f"  Accurate: {accurate_count}/{total} ({accurate_count/total*100:.1f}%)")
    
    # Calculate average error
    avg_error = sum(r['error'] for r in results) / len(results)
    print(f"📊 Average Error: {avg_error:.2f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR REAL IELTS-LIKE STRICTNESS:")
    print("=" * 60)
    
    if avg_error > 1.0:
        print("⚠️ System needs to be MORE STRICT")
        print("   - Increase weight on official IELTS criteria")
        print("   - Reduce leniency for low-quality essays")
        print("   - Implement stricter band score boundaries")
    elif avg_error < 0.5:
        print("✅ System is reasonably strict")
        print("   - Current strictness is appropriate")
        print("   - Consider fine-tuning for edge cases")
    else:
        print("🔄 System needs minor adjustments")
        print("   - Fine-tune weights for better accuracy")
        print("   - Adjust scoring boundaries")
    
    # Specific recommendations
    print(f"\n🎯 SPECIFIC RECOMMENDATIONS:")
    print("1. **For High-Quality Essays**: System should score closer to expected (7.5-8.5 range)")
    print("2. **For Low-Quality Essays**: System should be stricter (3.5-4.5 range)")
    print("3. **Weight Adjustment**: Use 70-80% Official IELTS + 20-30% ML")
    print("4. **Band Boundaries**: Implement stricter minimum requirements")

def main():
    """
    Main function to test and analyze strictness
    """
    print("🎯 EdPrep AI: Strictness Analysis for Real IELTS-like Assessment")
    print("=" * 70)
    
    # Test current strictness
    results = test_current_strictness()
    
    # Analyze results
    analyze_strictness(results)
    
    print(f"\n✅ Analysis Complete!")
    print("Next: Consider Google Colab Pro for advanced training with more data")

if __name__ == "__main__":
    main()
