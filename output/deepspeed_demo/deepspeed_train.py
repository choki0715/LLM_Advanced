"""DeepSpeed ZeRO FFT 학습 스크립트
Usage: torchrun --nproc_per_node=2 deepspeed_train.py --zero_stage 2
"""
import os
import sys
import json
import time
import argparse
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero_stage", type=int, default=2, choices=[2, 3])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    sample_texts = [
        "인공지능은 인간의 지능을 모방하여 학습, 추론, 판단 등의 작업을 수행하는 시스템입니다.",
        "머신러닝은 데이터로부터 패턴을 학습하는 알고리즘을 연구하는 분야입니다.",
        "딥러닝은 인공 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습합니다.",
        "트랜스포머는 셀프 어텐션 메커니즘을 핵심으로 하는 아키텍처입니다.",
        "LoRA는 적은 파라미터만 학습하면서도 좋은 성능을 달성할 수 있습니다.",
    ] * 10

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenizer(
        sample_texts, truncation=True, padding="max_length",
        max_length=256, return_tensors=None
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    dataset = Dataset.from_dict(tokenized)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()

    ds_config = f"./output/deepspeed_demo/ds_zero{args.zero_stage}_config.json"

    training_args = TrainingArguments(
        output_dir=f"./output/deepspeed_demo/zero{args.zero_stage}_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        deepspeed=ds_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    if trainer.is_world_process_zero():
        gpu_mem = []
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.max_memory_allocated(i) / 1024**3
            gpu_mem.append(f"{mem:.1f}GB")

        results = {
            "zero_stage": args.zero_stage,
            "training_loss": result.training_loss,
            "elapsed_seconds": round(elapsed, 1),
            "gpu_memory": gpu_mem,
            "num_gpus": torch.cuda.device_count(),
        }
        result_path = f"./output/deepspeed_demo/zero{args.zero_stage}_result.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ ZeRO-{args.zero_stage} 학습 완료!")
        print(f"📌 소요 시간: {elapsed:.1f}초")
        print(f"📌 Training Loss: {result.training_loss:.4f}")
        print(f"📌 GPU 메모리: {gpu_mem}")
        print(f"📌 결과 저장: {result_path}")

if __name__ == "__main__":
    main()
