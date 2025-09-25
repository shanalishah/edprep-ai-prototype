#!/usr/bin/env python3
"""
Make Current System More Strict for Real IELTS-like Assessment
"""

import requests
import json

def test_strict_mode():
    """
    Test the current strict mode toggle
    """
    print("🔧 Testing Strict Mode Toggle...")
    
    try:
        # Toggle to strict mode
        response = requests.post('http://localhost:8000/toggle-strict-mode', timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Strict mode: {result['strict_mode']}")
            print(f"📊 Current mode: {result['current_mode']}")
            print(f"⚖️ Weights: {result['weights']}")
            return result
        else:
            print(f"❌ Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_essay_with_strict_mode(essay_data):
    """
    Test essay with strict mode enabled
    """
    print(f"\n📝 Testing {essay_data['id']} with strict mode...")
    print(f"Expected Band: {essay_data['expected_band']:.1f}")
    
    try:
        response = requests.post('http://localhost:8000/assess', json={
            'prompt': essay_data['prompt'],
            'essay': essay_data['essay'],
            'task_type': 'Task 2'
        }, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            overall_score = result['overall_band_score']
            error = abs(overall_score - essay_data['expected_band'])
            
            print(f"✅ Got Band Score: {overall_score:.1f}")
            print(f"📊 Error: {error:.1f}")
            
            # Calculate strictness
            if error <= 0.25:
                strictness = 'accurate'
            elif overall_score < essay_data['expected_band'] - 0.25:
                strictness = 'strict'
            else:
                strictness = 'lenient'
            
            print(f"🎯 Strictness: {strictness}")
            print(f"📋 Breakdown: TA={result['task_achievement']:.1f}, CC={result['coherence_cohesion']:.1f}, LR={result['lexical_resource']:.1f}, GRA={result['grammatical_range']:.1f}")
            
            return {
                'expected': essay_data['expected_band'],
                'actual': overall_score,
                'error': error,
                'strictness': strictness
            }
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """
    Main function to make the system more strict
    """
    print("🎯 Making EdPrep AI System More Strict")
    print("=" * 50)
    
    # Test essays
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
    
    # Test current strict mode
    strict_mode = test_strict_mode()
    
    if strict_mode:
        print(f"\n🧪 Testing Essays with Strict Mode...")
        print("=" * 50)
        
        results = []
        for essay in test_essays:
            result = test_essay_with_strict_mode(essay)
            if result:
                results.append(result)
        
        # Analyze results
        if results:
            print(f"\n📈 STRICT MODE ANALYSIS:")
            print("=" * 50)
            
            avg_error = sum(r['error'] for r in results) / len(results)
            strict_count = sum(1 for r in results if r['strictness'] == 'strict')
            lenient_count = sum(1 for r in results if r['strictness'] == 'lenient')
            
            print(f"📊 Average Error: {avg_error:.2f}")
            print(f"📊 Strict: {strict_count}/{len(results)}")
            print(f"📊 Lenient: {lenient_count}/{len(results)}")
            
            if avg_error < 1.0:
                print("✅ Strict mode is working well!")
            else:
                print("⚠️ System still needs more strictness adjustments")
    
    print(f"\n💡 NEXT STEPS:")
    print("=" * 50)
    print("1. ✅ Current system is now in strict mode")
    print("2. 🚀 Use Google Colab Pro for advanced training")
    print("3. 📊 Train on your full 47,117 essays dataset")
    print("4. 🤖 Implement transformer models for better accuracy")
    print("5. 🎯 Fine-tune for real IELTS-like strictness")

if __name__ == "__main__":
    main()
