import yaml

with open('config/emotion_scoring_config.yaml', encoding='utf-8') as f:
    config = yaml.safe_load(f)

weights = config['criterion_weights']

print("Emotion Criterion Weights:")
print(f"  emotion_stability: {weights['emotion_stability']*100:.0f}%")
print(f"  emotion_content_alignment: {weights['emotion_content_alignment']*100:.0f}%")
print(f"  positive_ratio: {weights['positive_ratio']*100:.0f}%")
print(f"  negative_overload: {weights['negative_overload']*100:.0f}%")
print(f"\nTotal: {sum(weights.values())*100:.0f}%")
print(f"Final multiplier: {config['final_score_multiplier']}")
