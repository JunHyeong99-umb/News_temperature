import os
from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import numpy as np
import torch

# ===== 경로 설정 =====
TRAIN_PATH = "data/processed/train.jsonl"   # 전체 학습 데이터 사용
VALID_PATH = "data/processed/valid.jsonl"
OUT_DIR = "models/kosum-v1-fast-v2"

# 기존 학습된 모델을 teacher로 사용
TEACHER_MODEL_PATH = "models/kosum-v1"  # 기존 학습된 모델
BASE_MODEL_NAME = "gogamza/kobart-base-v2"
MAX_IN = 1024
MAX_OUT = 128

rouge = evaluate.load("rouge")


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

    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
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
    
    print("▶ 기존 학습된 모델(Teacher) 로드 중...")
    try:
        teacher_model = BartForConditionalGeneration.from_pretrained(TEACHER_MODEL_PATH)
        print(f"  ✓ Teacher 모델 로드 완료: {TEACHER_MODEL_PATH}")
    except Exception as e:
        print(f"  ⚠ Teacher 모델 로드 실패, 새로 학습합니다: {e}")
        teacher_model = None
    
    print("▶ Student 모델 로드 중 (기존 base 모델 사용)...")
    # 기존 base 모델을 그대로 사용 (경량화하지 않음)
    # 대신 추론 시 greedy decoding으로 빠르게 사용
    model = BartForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    
    # Teacher 모델이 있으면 가중치 일부 초기화
    if teacher_model is not None:
        print("  ✓ Teacher 모델의 가중치로 초기화 중...")
        try:
            # 인코더/디코더의 첫 몇 레이어 가중치 복사
            with torch.no_grad():
                # 인코더 레이어 복사
                for i in range(min(len(model.model.encoder.layers), len(teacher_model.model.encoder.layers))):
                    model.model.encoder.layers[i].load_state_dict(
                        teacher_model.model.encoder.layers[i].state_dict()
                    )
                # 디코더 레이어 복사
                for i in range(min(len(model.model.decoder.layers), len(teacher_model.model.decoder.layers))):
                    model.model.decoder.layers[i].load_state_dict(
                        teacher_model.model.decoder.layers[i].state_dict()
                    )
                # 임베딩 레이어 복사
                if model.model.shared.weight.shape == teacher_model.model.shared.weight.shape:
                    model.model.shared.weight.data = teacher_model.model.shared.weight.data.clone()
                print("  ✓ Teacher 모델 가중치 복사 완료")
        except Exception as e:
            print(f"  ⚠ 가중치 복사 실패 (무시하고 계속): {e}")

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
        save_steps=1000,
        logging_steps=200,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,  # 더 낮은 학습률로 fine-tuning
        num_train_epochs=3,  # 더 많은 epoch
        predict_with_generate=True,
        generation_max_length=MAX_OUT,
        fp16=False,
        save_total_limit=2,
        report_to=[],
        warmup_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
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

    print("▶ 학습 시작")
    print(f"  모델 크기: {sum(p.numel() for p in model.parameters())/1e6:.1f}M 파라미터")
    trainer.train()

    print("▶ 모델 저장 중...")
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"✅ 저장 완료: {OUT_DIR}")
    print("\n💡 이 모델은 기존 모델과 동일한 구조이지만, 추론 시 greedy decoding을 사용하면 빠르게 요약할 수 있습니다.")


if __name__ == "__main__":
    main()

