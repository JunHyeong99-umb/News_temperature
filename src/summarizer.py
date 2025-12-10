import os
import re
from pathlib import Path
from typing import List
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# =========================
#  모델 로드
# =========================
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "kosum-v1"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(_DEFAULT_MODEL_DIR))).expanduser().resolve()

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"MODEL_DIR '{MODEL_DIR}' does not exist. Set MODEL_DIR env var to a valid path.")

# 토크나이저는 base 모델에서 로드 (호환성 문제 방지)
BASE_MODEL_NAME = "gogamza/kobart-base-v2"
try:
    tok = PreTrainedTokenizerFast.from_pretrained(str(MODEL_DIR))
except Exception:
    # 토크나이저 로드 실패 시 base 모델에서 로드
    tok = PreTrainedTokenizerFast.from_pretrained(BASE_MODEL_NAME)

model = BartForConditionalGeneration.from_pretrained(str(MODEL_DIR))

# =========================
#  공통 유틸 함수
# =========================
def clean_repetition(text: str) -> str:
    """
    바로 붙어 있는 중복 단어 제거
    예: '정부가 정부가 국가 국가 단위' -> '정부가 국가 단위'
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text

    tokens = text.split(" ")
    cleaned = []
    for t in tokens:
        if cleaned and t == cleaned[-1]:
            # 바로 앞 단어와 같으면 스킵
            continue
        cleaned.append(t)

    return " ".join(cleaned)


def sentence_split(text: str) -> List[str]:
    """
    lookbehind 안 쓰고 안전하게 문장 분리
    - . ? ! 기준으로 문장 끝으로 본다.
    """
    text = text.replace("\n", " ").strip()
    if not text:
        return []

    # 구두점을 따로 분리
    parts = re.split(r"([.!?])", text)
    sentences = []
    buf = ""

    for p in parts:
        if not p:
            continue
        p = p.strip()
        if not p:
            continue

        if p in [".", "!", "?"]:
            # 문장 끝 구두점
            buf += p
            sentences.append(buf.strip())
            buf = ""
        else:
            # 일반 텍스트
            if buf:
                buf += " " + p
            else:
                buf = p

    if buf.strip():
        sentences.append(buf.strip())

    return sentences


def pick_first_sentences(text: str, n: int = 2) -> str:
    """
    요약 결과에서 앞 n개 문장만 사용 (뉴스 기사 1~2줄)
    """
    sents = sentence_split(text)
    return " ".join(sents[:n])


# =========================
#  1회 요약 (짧은 텍스트)
# =========================
def summarize_once(text: str, max_len: int = 300, fast: bool = True) -> str:
    """
    텍스트 한 번 요약 → 1~2문장 반환
    max_len은 '생성 토큰 상한'이지, 실제 문장 길이는 후처리로 1~2문장으로 잘린다.
    fast=True: greedy decoding 사용 (빠른 추론)
    fast=False: beam search 사용 (더 나은 품질)
    """
    text = (text or "").strip()
    if not text:
        return ""

    inputs = tok(text, max_length=1024, truncation=True, return_tensors="pt")
    # KoBART는 token_type_ids를 사용하지 않으므로 제거
    inputs.pop("token_type_ids", None)

    if fast:
        # 빠른 추론: greedy decoding (개선된 파라미터)
        output = model.generate(
            **inputs,
            max_length=max_len,
            min_length=30,
            num_beams=1,  # greedy decoding
            no_repeat_ngram_size=2,  # 반복 방지
            repetition_penalty=1.4,  # 반복 패널티 강화
            do_sample=False,
        )
    else:
        # 고품질 추론: beam search
        output = model.generate(
            **inputs,
            max_length=max_len,
            min_length=30,
            num_beams=4,
            no_repeat_ngram_size=4,
            repetition_penalty=1.15,
            length_penalty=1.0,
            early_stopping=True,
        )

    decoded = tok.decode(output[0], skip_special_tokens=True).strip()
    decoded = clean_repetition(decoded)
    
    # 추가 정리: 연속된 공백 제거
    decoded = re.sub(r'\s+', ' ', decoded).strip()
    
    # 불완전한 문장 제거 (마지막이 구두점으로 끝나지 않으면 제거)
    sentences = sentence_split(decoded)
    if sentences:
        # 마지막 문장이 구두점으로 끝나지 않으면 제거
        if sentences[-1] and sentences[-1][-1] not in ['.', '!', '?', '다', '요']:
            sentences = sentences[:-1]
        decoded = " ".join(sentences)

    # 최종 결과는 1~2문장만 사용
    final = pick_first_sentences(decoded, n=2)
    return final


# =========================
#  긴 문서 chunk 요약
# =========================
def summarize_long(text: str, chunk_chars: int = 1500, fast: bool = True) -> str:
    """
    긴 문서를 문장 단위로 잘라서:
      1) chunk_chars 기준으로 문장들을 묶어 여러 청크로 나누고,
      2) 각 청크를 summarize_once로 부분 요약한 뒤,
      3) 부분 요약들을 다시 합쳐 summarize_once로 재요약.
    fast: 빠른 추론 모드 사용 여부
    """
    text = (text or "").replace("\n", " ").strip()
    if not text:
        return ""

    sentences = sentence_split(text)
    if not sentences:
        return ""

    # 문장 묶어서 청크 만들기
    chunks: List[str] = []
    buf = ""

    for s in sentences:
        # 청크에 현재 문장을 추가했을 때 길이 제한 안 넘으면 같은 청크
        if len(buf) + len(s) + 1 <= chunk_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = s

    if buf:
        chunks.append(buf)

    # 청크가 1개뿐이면 그냥 한 번 요약
    if len(chunks) == 1:
        return summarize_once(chunks[0], max_len=300, fast=fast)

    # 1단계: 각 청크를 짧게 요약
    partial_summaries: List[str] = [
        summarize_once(c, max_len=250, fast=fast) for c in chunks
    ]

    # 2단계: 부분 요약들을 합쳐서 다시 한번 요약
    combined = " ".join(partial_summaries)
    final = summarize_once(combined, max_len=300, fast=fast)

    return final


# =========================
#  외부에서 쓰는 메인 함수
# =========================
def summarize(text: str, chunk_chars: int = 1500, max_len: int = 300, fast: bool = True) -> str:
    """
    - 짧은 문장: summarize_once 사용
    - 긴 문장: summarize_long(chunk) 사용
    fast=True: 빠른 추론 모드 (greedy decoding, 기본값)
    fast=False: 고품질 모드 (beam search)
    외부에서는 이 함수만 호출해도 됨.
    """
    text = (text or "").strip()
    if not text:
        return ""

    if len(text) <= chunk_chars:
        return summarize_once(text, max_len=max_len, fast=fast)
    else:
        return summarize_long(text, chunk_chars=chunk_chars, fast=fast)
