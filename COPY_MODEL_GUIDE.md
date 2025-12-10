# 모델 복사 가이드

## 필요한 파일 목록

다른 곳에서 모델을 사용하려면 다음 파일들이 필요합니다:

### 필수 파일 (모델 디렉토리: `models/kosum-v1/`)

```
models/kosum-v1/
├── config.json                    # 모델 설정 파일 (필수)
├── model.safetensors             # 모델 가중치 파일 (필수, 가장 큼)
├── tokenizer.json                 # 토크나이저 파일 (필수)
├── tokenizer_config.json         # 토크나이저 설정 (필수)
├── special_tokens_map.json       # 특수 토큰 매핑 (필수)
└── generation_config.json        # 생성 설정 (선택사항, 있으면 좋음)
```

### 불필요한 파일 (복사 안 해도 됨)

- `optimizer.pt` - 학습 최적화 상태 (추론 시 불필요)
- `rng_state.pth` - 랜덤 시드 상태 (추론 시 불필요)
- `scheduler.pt` - 학습 스케줄러 (추론 시 불필요)
- `trainer_state.json` - 학습 상태 (추론 시 불필요)
- `training_args.bin` - 학습 인자 (추론 시 불필요)
- `checkpoint-313/` - 체크포인트 디렉토리 (추론 시 불필요)

## 복사 방법

### 방법 1: 필요한 파일만 수동 복사

```bash
# 새 디렉토리 생성
mkdir my_model

# 필수 파일만 복사
cp models/kosum-v1/config.json my_model/
cp models/kosum-v1/model.safetensors my_model/
cp models/kosum-v1/tokenizer.json my_model/
cp models/kosum-v1/tokenizer_config.json my_model/
cp models/kosum-v1/special_tokens_map.json my_model/
cp models/kosum-v1/generation_config.json my_model/
```

### 방법 2: Python 스크립트로 복사

```python
import shutil
from pathlib import Path

# 원본 모델 경로
source_dir = Path("models/kosum-v1")
# 복사할 경로
target_dir = Path("my_model")

# 필요한 파일 목록
required_files = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
]

# 디렉토리 생성
target_dir.mkdir(exist_ok=True)

# 파일 복사
for file in required_files:
    source_file = source_dir / file
    if source_file.exists():
        shutil.copy2(source_file, target_dir / file)
        print(f"✓ {file} 복사 완료")
    else:
        print(f"⚠ {file} 없음 (선택사항)")

print(f"\n✅ 모델 복사 완료: {target_dir}")
```

### 방법 3: 전체 디렉토리 압축 (Windows)

```powershell
# PowerShell에서 실행
Compress-Archive -Path models\kosum-v1\config.json,models\kosum-v1\model.safetensors,models\kosum-v1\tokenizer.json,models\kosum-v1\tokenizer_config.json,models\kosum-v1\special_tokens_map.json,models\kosum-v1\generation_config.json -DestinationPath kosum-v1-model.zip
```

## 다른 곳에서 사용하기

### Python 코드 예시

```python
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# 모델 경로 지정
model_path = "path/to/my_model"  # 또는 "models/kosum-v1"

# 모델과 토크나이저 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# 사용 예시
text = "요약할 텍스트..."
inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
output = model.generate(**inputs, max_length=128, num_beams=1)
summary = tokenizer.decode(output[0], skip_special_tokens=True)
print(summary)
```

## 파일 크기 확인

```bash
# Windows PowerShell
Get-ChildItem models\kosum-v1\*.safetensors,models\kosum-v1\*.json | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB, 2)}}

# Linux/Mac
ls -lh models/kosum-v1/*.safetensors models/kosum-v1/*.json
```

## 주의사항

1. **model.safetensors** 파일이 가장 큽니다 (수백 MB ~ 수 GB)
2. 토크나이저 파일들(`tokenizer.json`)도 꽤 큽니다 (수십 MB)
3. 네트워크로 전송할 때는 압축을 사용하는 것이 좋습니다
4. 모델 경로에 공백이나 특수문자가 있으면 문제가 될 수 있습니다

