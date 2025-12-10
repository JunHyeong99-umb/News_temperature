"""
원본 모델과 Fast 모델의 속도 비교 테스트
사용법: python test_compare_speed.py
"""
from src.summarizer import summarize as summarize_original
import time

def test_original_model():
    """원본 모델 테스트"""
    print("=" * 70)
    print("원본 모델 테스트 (Beam Search vs Greedy Decoding)")
    print("=" * 70)
    
    test_text = """
    최근 환율을 비롯한 각종 대외 변수로 코스피지수 변동성이 극대화된 가운데 한국의 개인투자자들이 장밋빛 전망을 믿고 빚까지 내 투자에 나서자 주요 외신들도 이를 불안하게 바라보고 있다. 
    글로벌 '큰손'들이 집결한 미국에서 악재가 발생해도 개인들의 매수로 지수 하락을 방어하는 형국이라 위험도가 더 높아졌다는 진단도 나온다. 
    특히 최근 한국의 주식 투자 열풍이 집값 급등과 연관돼 있다고 보는 외신도 나오고 있어 눈길을 끌고 있다. 
    정부의 부동산 대책이 잇따라 실패하고 집값이 천정부지로 치솟자 상대적 박탈감을 느끼고 이를 따라잡기 위한 사람들이 무리수를 두는 게 아니냐는 시각이다.
    블룼버그통신은 지난 12일(현지 시간) 최근 코스피시장에 대해 "한국 주식 변동에 대한 베팅이 급증하면서 올해 가장 좋은 성과를 거둔 시장의 랠리에 대한 우려가 커지고 있다"고 분석했다.
    """
    
    print(f"\n원문 길이: {len(test_text)} 글자\n")
    
    # Beam Search 모드
    print("-" * 70)
    print("[1] 원본 모델 - Beam Search (고품질 모드)")
    print("-" * 70)
    start = time.time()
    summary_beam = summarize_original(test_text, fast=False)
    time_beam = time.time() - start
    print(f"요약: {summary_beam}")
    print(f"소요 시간: {time_beam:.2f}초\n")
    
    # Greedy Decoding 모드
    print("-" * 70)
    print("[2] 원본 모델 - Greedy Decoding (빠른 모드)")
    print("-" * 70)
    start = time.time()
    summary_greedy = summarize_original(test_text, fast=True)
    time_greedy = time.time() - start
    print(f"요약: {summary_greedy}")
    print(f"소요 시간: {time_greedy:.2f}초\n")
    
    # 성능 비교
    print("=" * 70)
    print("성능 비교")
    print("=" * 70)
    print(f"Beam Search:     {time_beam:.2f}초")
    print(f"Greedy:         {time_greedy:.2f}초")
    if time_greedy > 0:
        speedup = time_beam / time_greedy
        print(f"속도 향상:      {speedup:.2f}x 빠름")
    print("=" * 70)


def test_fast_model():
    """Fast 모델 테스트"""
    try:
        from src.summarizer_fast import summarize as summarize_fast
        
        print("\n\n" + "=" * 70)
        print("Fast 모델 테스트 (경량화된 모델)")
        print("=" * 70)
        
        test_text = """
        최근 환율을 비롯한 각종 대외 변수로 코스피지수 변동성이 극대화된 가운데 한국의 개인투자자들이 장밋빛 전망을 믿고 빚까지 내 투자에 나서자 주요 외신들도 이를 불안하게 바라보고 있다. 
        글로벌 '큰손'들이 집결한 미국에서 악재가 발생해도 개인들의 매수로 지수 하락을 방어하는 형국이라 위험도가 더 높아졌다는 진단도 나온다. 
        특히 최근 한국의 주식 투자 열풍이 집값 급등과 연관돼 있다고 보는 외신도 나오고 있어 눈길을 끌고 있다. 
        정부의 부동산 대책이 잇따라 실패하고 집값이 천정부지로 치솟자 상대적 박탈감을 느끼고 이를 따라잡기 위한 사람들이 무리수를 두는 게 아니냐는 시각이다.
        블룼버그통신은 지난 12일(현지 시간) 최근 코스피시장에 대해 "한국 주식 변동에 대한 베팅이 급증하면서 올해 가장 좋은 성과를 거둔 시장의 랠리에 대한 우려가 커지고 있다"고 분석했다.
        """
        
        print(f"\n원문 길이: {len(test_text)} 글자\n")
        
        start = time.time()
        summary_fast = summarize_fast(test_text)
        time_fast = time.time() - start
        
        print(f"요약: {summary_fast}")
        print(f"소요 시간: {time_fast:.2f}초")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nFast 모델 로드 실패: {e}")
        print("Fast 모델이 아직 학습되지 않았거나 경로가 잘못되었습니다.")


if __name__ == "__main__":
    try:
        test_original_model()
        test_fast_model()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

