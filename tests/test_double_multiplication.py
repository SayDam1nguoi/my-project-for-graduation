"""
Test Double Multiplication Issue

Kiểm tra xem có bị tính chồng chéo không:
- UI "Nhận diện cảm xúc" tính điểm × final_score_multiplier
- UI "Tổng kết điểm" nhân thêm × weight (5%)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


def test_emotion_scoring_flow():
    """Test toàn bộ flow tính điểm emotion."""
    
    print("=" * 70)
    print("KIỂM TRA TÍNH CHỒNG CHÉO ĐIỂM")
    print("=" * 70)
    print()
    
    # ===== BƯỚC 1: Emotion Scoring Engine =====
    print("BƯỚC 1: Emotion Scoring Engine")
    print("-" * 70)
    
    # Load config
    with open('config/emotion_scoring_config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    criterion_weights = config['criterion_weights']
    final_multiplier = config['final_score_multiplier']
    
    print("Config:")
    print(f"  emotion_stability weight: {criterion_weights['emotion_stability']*100:.0f}%")
    print(f"  final_score_multiplier: {final_multiplier}")
    print()
    
    # Giả sử điểm emotion_stability = 8.5/10
    emotion_stability_score = 8.5
    
    print("Tính toán:")
    print(f"  emotion_stability_score = {emotion_stability_score}/10")
    print(f"  weighted_sum = {emotion_stability_score} × {criterion_weights['emotion_stability']} = {emotion_stability_score * criterion_weights['emotion_stability']}")
    
    # Apply final multiplier
    emotion_engine_output = emotion_stability_score * criterion_weights['emotion_stability'] * final_multiplier
    print(f"  emotion_engine_output = {emotion_stability_score * criterion_weights['emotion_stability']} × {final_multiplier} = {emotion_engine_output}")
    print()
    print(f"✅ Output từ Emotion Engine: {emotion_engine_output}/10")
    print()
    
    # ===== BƯỚC 2: ScoreManager =====
    print("BƯỚC 2: ScoreManager (GUI)")
    print("-" * 70)
    
    # Điểm này được set vào ScoreManager
    print(f"ScoreManager.set_emotion_score({emotion_engine_output})")
    print(f"  → self.emotion_score = {emotion_engine_output}/10")
    print()
    
    # ===== BƯỚC 3: Calculate Total Score =====
    print("BƯỚC 3: Calculate Total Score (UI Tổng Kết)")
    print("-" * 70)
    
    # Hiện tại: emotion weight = 100% (đã đơn giản hóa)
    emotion_weight_percent = 100.0
    
    print("Công thức:")
    print(f"  Total = emotion_score × (emotion_weight / 100)")
    print(f"  Total = {emotion_engine_output} × ({emotion_weight_percent} / 100)")
    print(f"  Total = {emotion_engine_output} × {emotion_weight_percent/100}")
    
    total_score = emotion_engine_output * (emotion_weight_percent / 100.0)
    print(f"  Total = {total_score}")
    print()
    print(f"✅ Điểm tổng cuối cùng: {total_score}/10")
    print()
    
    # ===== PHÂN TÍCH =====
    print("=" * 70)
    print("PHÂN TÍCH")
    print("=" * 70)
    print()
    
    print("Flow hiện tại:")
    print(f"  1. Emotion Stability: {emotion_stability_score}/10")
    print(f"  2. × criterion_weight ({criterion_weights['emotion_stability']*100:.0f}%): {emotion_stability_score * criterion_weights['emotion_stability']}/10")
    print(f"  3. × final_multiplier ({final_multiplier}): {emotion_engine_output}/10")
    print(f"  4. × emotion_weight ({emotion_weight_percent}%): {total_score}/10")
    print()
    
    # Kiểm tra có bị nhân 2 lần không
    if final_multiplier == 1.0 and emotion_weight_percent == 100.0:
        print("✅ KHÔNG BỊ TÍNH CHỒNG CHÉO!")
        print()
        print("Lý do:")
        print("  - final_score_multiplier = 1.0 (không thay đổi điểm)")
        print("  - emotion_weight = 100% (= ×1.0)")
        print("  - Điểm cuối = Điểm ban đầu")
        print()
        print(f"Kết quả: {emotion_stability_score}/10 → {total_score}/10 ✅")
    else:
        print("⚠️  CÓ THỂ BỊ TÍNH CHỒNG CHÉO!")
        print()
        print("Nếu:")
        print(f"  - final_score_multiplier = {final_multiplier}")
        print(f"  - emotion_weight = {emotion_weight_percent}%")
        print()
        print(f"Thì điểm sẽ bị nhân: {emotion_stability_score} × {final_multiplier} × {emotion_weight_percent/100} = {total_score}")
    
    print()
    print("=" * 70)
    
    # Test với hệ thống cũ (5%)
    print()
    print("KIỂM TRA HỆ THỐNG CŨ (Emotion = 5%)")
    print("=" * 70)
    print()
    
    old_emotion_weight = 5.0
    
    print("Nếu emotion_weight = 5%:")
    print(f"  Total = {emotion_engine_output} × ({old_emotion_weight} / 100)")
    old_total = emotion_engine_output * (old_emotion_weight / 100.0)
    print(f"  Total = {old_total}/10")
    print()
    
    if final_multiplier == 0.5:
        print("⚠️  VẤN ĐỀ PHÁT HIỆN!")
        print()
        print("Với final_multiplier = 0.5:")
        print(f"  1. Emotion Stability: {emotion_stability_score}/10")
        print(f"  2. × 0.5 (multiplier): {emotion_stability_score * 0.5}/10")
        print(f"  3. × 5% (weight): {emotion_stability_score * 0.5 * 0.05}/10")
        print()
        print("Điểm emotion chỉ đóng góp 0.21 điểm vào tổng 10 điểm!")
        print("→ Quá nhỏ, không hợp lý!")
    else:
        print("✅ Với final_multiplier = 1.0:")
        print(f"  1. Emotion Stability: {emotion_stability_score}/10")
        print(f"  2. × 1.0 (multiplier): {emotion_stability_score}/10")
        print(f"  3. × 5% (weight): {emotion_stability_score * 0.05}/10")
        print()
        print(f"Điểm emotion đóng góp {emotion_stability_score * 0.05:.2f} điểm vào tổng 10 điểm")
        print("→ Hợp lý! ✅")


if __name__ == "__main__":
    test_emotion_scoring_flow()
