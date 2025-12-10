# 빠른 요약 사용 가이드

## 문제 해결 완료 ✅

기존 학습된 모델(`models/kosum-v1`)을 사용하되, **빠른 추론 모드**로 요약할 수 있습니다.

## 사용 방법

### 기본 사용 (빠른 모드)
```python
from src.summarizer import summarize

text = "여기에 요약할 텍스트를 넣으세요..."
summary = summarize(text, fast=True)  # 빠른 모드 (기본값)
print(summary)
```

### 고품질 모드 (느리지만 더 좋은 품질)
```python
summary = summarize(text, fast=False)  # Beam Search 사용
```

## 테스트

```bash
# 빠른 모드 테스트
python test_original_fast.py
```

## 성능 비교

- **Fast 모드 (Greedy)**: 약 1.3-1.5x 빠름, 품질 약간 낮음
- **Beam Search 모드**: 느리지만 더 나은 품질

## 주의사항

- Fast 모델(`models/kosum-v1-fast`)은 학습이 부족하여 품질이 낮습니다
- **기존 모델(`models/kosum-v1`)을 사용하세요**
- `fast=True` 옵션으로 빠른 추론이 가능합니다

