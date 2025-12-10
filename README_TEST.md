# 모델 테스트 가이드

## Fast 모델 테스트

### 1. Fast 모델 단독 테스트
```bash
python test_fast_model.py
```

이 스크립트는 Fast 모델의 성능을 다양한 길이의 텍스트로 테스트합니다.

### 2. 원본 모델과 속도 비교
```bash
python test_compare_speed.py
```

이 스크립트는 원본 모델의 Beam Search와 Greedy Decoding 모드를 비교하고, Fast 모델과도 비교합니다.

## 모델 정보

### 원본 모델 (kosum-v1)
- 경로: `models/kosum-v1`
- 모델 크기: 약 140M 파라미터
- 추론 방식:
  - `fast=False`: Beam Search (고품질, 느림)
  - `fast=True`: Greedy Decoding (빠름)

### Fast 모델 (kosum-v1-fast)
- 경로: `models/kosum-v1-fast`
- 모델 크기: 약 45.8M 파라미터 (약 1/3 크기)
- 추론 방식: Greedy Decoding (빠른 추론에 최적화)

## 사용 예시

### 원본 모델 사용
```python
from src.summarizer import summarize

# 빠른 모드 (Greedy Decoding)
summary = summarize(text, fast=True)

# 고품질 모드 (Beam Search)
summary = summarize(text, fast=False)
```

### Fast 모델 사용
```python
from src.summarizer_fast import summarize

# Fast 모델은 항상 빠른 추론 모드
summary = summarize(text)
```

## 성능 비교

- **원본 모델 (Beam Search)**: 가장 높은 품질, 가장 느림
- **원본 모델 (Greedy)**: 중간 품질, 중간 속도
- **Fast 모델**: 빠른 속도, 경량화된 모델

Fast 모델은 경량화되어 있어 추론 속도가 빠르며, 메모리 사용량도 적습니다.

