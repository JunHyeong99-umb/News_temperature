import os
from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BartConfig,
)
import evaluate
import numpy as np

# ===== 경로 설정 =====
TRAIN_PATH = "data/processed/train_5k.jsonl"   # 연기 테스트용 5k
VALID_PATH = "data/processed/valid.jsonl"
OUT_DIR = "models/kosum-v1-fast"

# 기존 모델에서 경량화된 버전 생성
BASE_MODEL_NAME = "gogamza/kobart-base-v2"
MAX_IN = 1024
MAX_OUT = 128

rouge = evaluate.load("rouge")


def create_lightweight_config(base_config):
    """
    기존 BART 설정에서 경량화된 설정 생성
    - 레이어 수 감소 (6 -> 4)
    - Hidden dimension 감소 (768 -> 512)
    - FFN dimension 감소 (3072 -> 2048)
    - Attention heads 감소 (16 -> 8)
    """
    config = BartConfig.from_pretrained(base_config)
    
    # 경량화 설정
    config.encoder_layers = 4  # 6 -> 4
    config.decoder_layers = 4   # 6 -> 4
    config.d_model = 512       # 768 -> 512
    config.encoder_ffn_dim = 2048  # 3072 -> 2048
    config.decoder_ffn_dim = 2048  # 3072 -> 2048
    config.encoder_attention_heads = 8  # 16 -> 8
    config.decoder_attention_heads = 8  # 16 -> 8
    
    return config


def load_jsonl(train_path, valid_path):
    ds_tr = load_dataset("json", data_files=train_path, split="train")
    ds_va = load_dataset("json", data_files=valid_path, split="train")
    return {"train": ds_tr, "test": ds_va}


def preprocess_function(examples, tok):
    inputs = examples["document"]
    targets = examples["summary"]

    model_inputs = tok(
        inputs,
        max_length=MAX_IN,
        truncation=True,
    )
    with tok.as_target_tokenizer():
        labels = tok(
            targets,
            max_length=MAX_OUT,
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tok):
    preds, labels = eval_pred

    # 일부 환경에서는 logits로 들어와서 argmax 필요
    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)

    # -100 제거
    labels = [[l for l in label if l != -100] for label in labels]
    decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=[p.strip() for p in decoded_preds],
        references=[l.strip() for l in decoded_labels],
        use_stemmer=True,
    )
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("▶ 토크나이저 로드 중...")
    tok = PreTrainedTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    
    print("▶ 경량화된 모델 설정 생성 중...")
    lightweight_config = create_lightweight_config(BASE_MODEL_NAME)
    
    print("▶ 경량화된 모델 초기화 중...")
    # 경량화된 설정으로 새 모델 생성
    model = BartForConditionalGeneration(lightweight_config)
    
    # 기존 모델의 임베딩 레이어 가중치 복사 (vocab size가 같다면)
    try:
        base_model = BartForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
        # 임베딩 레이어만 복사 (크기가 맞는 경우)
        if model.model.shared.weight.shape == base_model.model.shared.weight.shape:
            model.model.shared.weight.data = base_model.model.shared.weight.data.clone()
            print("  ✓ 임베딩 레이어 가중치 복사 완료")
    except Exception as e:
        print(f"  ⚠ 임베딩 레이어 복사 실패 (무시하고 계속): {e}")

    print("▶ 데이터 로드 중...")
    ds = load_jsonl(TRAIN_PATH, VALID_PATH)

    print("▶ 토크나이즈 중...")
    tokenized_train = ds["train"].map(
        lambda e: preprocess_function(e, tok),
        batched=True,
        remove_columns=ds["train"].column_names,
    )
    tokenized_val = ds["test"].map(
        lambda e: preprocess_function(e, tok),
        batched=True,
        remove_columns=ds["test"].column_names,
    )

    collator = DataCollatorForSeq2Seq(tok, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        save_steps=500,
        logging_steps=100,
        per_device_train_batch_size=4,  # 경량 모델이므로 배치 크기 증가 가능
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=2,  # 경량 모델이므로 더 많은 epoch 가능
        predict_with_generate=True,
        generation_max_length=MAX_OUT,
        fp16=False,  # 안정성을 위해 끔 (필요하면 True)
        save_total_limit=2,
        report_to=[],
        warmup_steps=100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=lambda p: compute_metrics(p, tok),
    )

    print("▶ 학습 시작 (경량화된 모델)")
    print(f"  모델 크기: {sum(p.numel() for p in model.parameters())/1e6:.1f}M 파라미터")
    trainer.train()

    print("▶ 모델 저장 중...")
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"✅ 저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()

