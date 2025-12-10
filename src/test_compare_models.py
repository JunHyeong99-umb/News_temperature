from summarizer import summarize as summarize_original
from summarizer_fast import summarize as summarize_fast
import time

if __name__ == "__main__":
    # 테스트용 텍스트
    test_text = """
    최근 환율을 비롯한 각종 대외 변수로 코스피지수 변동성이 극대화된 가운데 한국의 개인투자자들이 장밋빛 전망을 믿고 빚까지 내 투자에 나서자 주요 외신들도 이를 불안하게 바라보고 있다. 글로벌 '큰손'들이 집결한 미국에서 악재가 발생해도 개인들의 매수로 지수 하락을 방어하는 형국이라 위험도가 더 높아졌다는 진단도 나온다. 특히 최근 한국의 주식 투자 열풍이 집값 급등과 연관돼 있다고 보는 외신도 나오고 있어 눈길을 끌고 있다. 정부의 부동산 대책이 잇따라 실패하고 집값이 천정부지로 치솟자 상대적 박탈감을 느끼고 이를 따라잡기 위한 사람들이 무리수를 두는 게 아니냐는 시각이다.
    블룼버그통신은 지난 12일(현지 시간) 최근 코스피시장에 대해 "한국 주식 변동에 대한 베팅이 급증하면서 올해 가장 좋은 성과를 거둔 시장의 랠리에 대한 우려가 커지고 있다"고 분석했다. 블룼버그통신은 "코스피200 변동성지수가 도널드 트럼프 미국 대통령의 관세로 촉발된 4월 시장 침체 당시 수준으로 뛰어올랐다"며 "이는 다른 시장은 상대적으로 평온한 상태에서 드물게 벗어난 급등 현상"이라고 지적했다.
    """

    print("=" * 70)
    print("모델 비교 테스트")
    print("=" * 70)
    print(f"\n원문 길이: {len(test_text)} 글자\n")

    # 원본 모델 테스트 (fast=False: beam search)
    print("-" * 70)
    print("1. 원본 모델 (Beam Search)")
    print("-" * 70)
    start_time = time.time()
    summary_original = summarize_original(test_text, fast=False)
    time_original = time.time() - start_time
    print(f"요약: {summary_original}")
    print(f"소요 시간: {time_original:.2f}초\n")

    # 원본 모델 fast 모드 테스트
    print("-" * 70)
    print("2. 원본 모델 (Greedy Decoding - Fast 모드)")
    print("-" * 70)
    start_time = time.time()
    summary_original_fast = summarize_original(test_text, fast=True)
    time_original_fast = time.time() - start_time
    print(f"요약: {summary_original_fast}")
    print(f"소요 시간: {time_original_fast:.2f}초\n")

    # Fast 모델 테스트
    print("-" * 70)
    print("3. Fast 모델 (경량화된 모델)")
    print("-" * 70)
    try:
        start_time = time.time()
        summary_fast = summarize_fast(test_text)
        time_fast = time.time() - start_time
        print(f"요약: {summary_fast}")
        print(f"소요 시간: {time_fast:.2f}초\n")
    except Exception as e:
        print(f"오류 발생: {e}\n")

    # 성능 비교
    print("=" * 70)
    print("성능 비교")
    print("=" * 70)
    print(f"원본 모델 (Beam Search):     {time_original:.2f}초")
    print(f"원본 모델 (Greedy):          {time_original_fast:.2f}초 ({time_original/time_original_fast:.2f}x 빠름)")
    if 'time_fast' in locals():
        print(f"Fast 모델 (경량화):         {time_fast:.2f}초 ({time_original/time_fast:.2f}x 빠름)")
    print("=" * 70)

