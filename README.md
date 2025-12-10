# News Temperature - 텍스트 요약 모델

한국어 뉴스 기사 요약을 위한 BART 기반 모델 프로젝트입니다.

## 주요 기능

- 한국어 텍스트 요약 (KoBART 기반)
- 빠른 추론 모드 (Greedy Decoding)
- 고품질 추론 모드 (Beam Search)
- 긴 문서 청크 단위 요약 지원

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 기본 사용

```python
from src.summarizer import summarize

text = "요약할 텍스트를 여기에 입력하세요..."
summary = summarize(text, fast=True)  # 빠른 모드
print(summary)
```

### 고품질 모드

```python
summary = summarize(text, fast=False)  # Beam Search 사용
```

## 모델 학습

### 기본 모델 학습
```bash
python src/train_summarizer.py
```

### Fast 모델 학습 (경량화)
```bash
python src/train_summarizer_fast.py
```

## 테스트

```bash
# Fast 모드 테스트
python test_original_fast.py

# 모델 비교 테스트
python test_compare_speed.py
```

## 모델 파일

학습된 모델은 `models/kosum-v1/` 디렉토리에 저장됩니다.

다른 곳에서 사용하려면 다음 파일들이 필요합니다:
- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `generation_config.json`

자세한 내용은 `COPY_MODEL_GUIDE.md`를 참고하세요.

## 프로젝트 구조

```
.
├── src/
│   ├── summarizer.py          # 메인 요약 모듈
│   ├── summarizer_fast.py      # Fast 모델용 요약 모듈
│   ├── train_summarizer.py     # 학습 스크립트
│   └── preprocess_aihub_zip.py # 데이터 전처리
├── models/
│   └── kosum-v1/              # 학습된 모델
├── data/
│   └── processed/             # 전처리된 데이터
└── test_*.py                   # 테스트 스크립트들
```

## 라이선스

MIT License

