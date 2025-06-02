# -*- coding: utf-8 -*-
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)

def load_jsonl(file_path):
    """加载JSONL格式的数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    # 转换为问答格式
    samples = []
    for item in data:
        prompt = f"年龄: {item['age']}，性别: {item['gender']}，症状: {','.join(item['symptoms'])}。请给出诊断结果。"
        answer = item['diagnosis']
        samples.append({'prompt': prompt, 'answer': answer})
    return samples

def main():
    # 1. 加载数据
    print("正在加载数据...")
    data = load_jsonl('reduced_medical_data.jsonl')
    dataset = Dataset.from_list(data)
    print(f"数据加载完成，共 {len(dataset)} 条样本")

    # 2. 加载模型和分词器
    print("正在加载模型和分词器...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float32,  # 使用float32
        device_map="mps"  # 使用MPS
    )
    print("模型加载完成")

    # 3. 数据预处理
    print("正在处理数据...")
    def preprocess(example):
        text = f"{example['prompt']}\n答：{example['answer']}"
        return tokenizer(text, truncation=True, max_length=512)

    tokenized_dataset = dataset.map(preprocess, remove_columns=['prompt', 'answer'])
    print("数据处理完成")

    # 4. 设置训练参数
    training_args = TrainingArguments(
        output_dir="./qwen2.5-medical-finetune",
        per_device_train_batch_size=1,  # 减小batch size
        num_train_epochs=2,
        save_steps=100,
        logging_steps=10,
        learning_rate=1e-5,  # 降低学习率
        report_to="none",
        gradient_accumulation_steps=8,  # 增加梯度累积步数
        warmup_steps=100,
        weight_decay=0.01,
        use_mps_device=True,  # 使用MPS
        fp16=False,  # 禁用fp16
    )

    # 5. 设置训练器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 6. 开始训练
    print("开始训练...")
    trainer.train()
    print("训练完成！")

    # 7. 保存模型
    print("正在保存模型...")
    trainer.save_model("./qwen2.5-medical-finetune-final")
    tokenizer.save_pretrained("./qwen2.5-medical-finetune-final")
    print("模型保存完成！")

if __name__ == "__main__":
    main() 