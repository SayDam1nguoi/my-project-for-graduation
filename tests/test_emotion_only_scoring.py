"""
Test Emotion-Only Scoring System

Verify that the system only evaluates Emotion Stability.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.overall_interview_scorer import OverallInterviewScorer


def test_emotion_only_weights():
    """Test that all weights are set to emotion only."""
    
    print("=" * 60)
    print("TESTING EMOTION-ONLY SCORING SYSTEM")
    print("=" * 60)
    print()
    
    # Test default weights
    scorer = OverallInterviewScorer(position_type='default')
    
    print("1. DEFAULT WEIGHTS:")
    print(f"   Emotion: {scorer.weights['emotion']*100:.0f}%")
    print(f"   Focus: {scorer.weights['focus']*100:.0f}%")
    print(f"   Clarity: {scorer.weights['clarity']*100:.0f}%")
    print(f"   Content: {scorer.weights['content']*100:.0f}%")
    print()
    
    # Verify
    assert scorer.weights['emotion'] == 1.0, "Emotion should be 100%"
    assert scorer.weights['focus'] == 0.0, "Focus should be 0%"
    assert scorer.weights['clarity'] == 0.0, "Clarity should be 0%"
    assert scorer.weights['content'] == 0.0, "Content should be 0%"
    print("   ✅ Default weights correct!")
    print()
    
    # Test all position types
    positions = ['technical', 'sales', 'customer_service', 'management']
    
    for position in positions:
        scorer = OverallInterviewScorer(position_type=position)
        print(f"2. {position.upper()} WEIGHTS:")
        print(f"   Emotion: {scorer.weights['emotion']*100:.0f}%")
        print(f"   Focus: {scorer.weights['focus']*100:.0f}%")
        print(f"   Clarity: {scorer.weights['clarity']*100:.0f}%")
        print(f"   Content: {scorer.weights['content']*100:.0f}%")
        
        assert scorer.weights['emotion'] == 1.0, f"{position}: Emotion should be 100%"
        assert scorer.weights['focus'] == 0.0, f"{position}: Focus should be 0%"
        assert scorer.weights['clarity'] == 0.0, f"{position}: Clarity should be 0%"
        assert scorer.weights['content'] == 0.0, f"{position}: Content should be 0%"
        print(f"   ✅ {position} weights correct!")
        print()
    
    print("=" * 60)
    print("3. TESTING SCORE CALCULATION:")
    print("=" * 60)
    print()
    
    # Test score calculation
    scorer = OverallInterviewScorer()
    
    # Test case: emotion=8.5, others=0 (should be ignored)
    score = scorer.calculate_score(
        emotion_score=8.5,
        focus_score=7.0,      # Should be ignored
        clarity_score=9.0,    # Should be ignored
        content_score=6.0     # Should be ignored
    )
    
    print(f"Input scores:")
    print(f"  Emotion: 8.5/10")
    print(f"  Focus: 7.0/10 (ignored)")
    print(f"  Clarity: 9.0/10 (ignored)")
    print(f"  Content: 6.0/10 (ignored)")
    print()
    print(f"Calculation:")
    print(f"  Total = Emotion × 100%")
    print(f"  Total = 8.5 × 1.0")
    print(f"  Total = {score.total_score}/10")
    print()
    
    # Verify total score equals emotion score
    assert abs(score.total_score - 8.5) < 0.01, f"Total should be 8.5, got {score.total_score}"
    print(f"✅ Score calculation correct!")
    print(f"   Total score = Emotion score = {score.total_score}/10")
    print()
    
    # Test rating
    print(f"Rating: {score.overall_rating}")
    print()
    
    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("Summary:")
    print("  - All weights set to Emotion: 100%")
    print("  - Focus, Clarity, Content: 0% (ignored)")
    print("  - Total score = Emotion score")
    print("  - System only evaluates Emotion Stability")


if __name__ == "__main__":
    test_emotion_only_weights()
