import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import json
import os
from tqdm import tqdm
import torch.nn.functional as F

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据集类
class QADataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.data = data['questions']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # 构建输入文本，添加特殊标记
        input_text = f"<|im_start|>user\nQuestion: {item['question']}\n<|im_end|>\n<|im_start|>assistant\nAnswer:"
        target_text = item['answer'] + "\n<|im_end|>"
        
        # 对输入和目标文本进行编码
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# 知识蒸馏损失函数
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=1.0):
    """计算蒸馏损失"""
    # 原始交叉熵损失
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
    
    # 蒸馏损失（KL散度）
    student_log_softmax = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_softmax = F.softmax(teacher_logits / temperature, dim=-1)
    distill_loss = F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (temperature ** 2)
    
    # 组合损失
    total_loss = alpha * ce_loss + (1 - alpha) * distill_loss
    return total_loss

def create_student_model():
    """创建学生模型，使用 Qwen 0.5B 作为基础"""
    # 加载 Qwen 0.5B 模型
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n学生模型配置:")
    print(f"总参数量: {total_params:,}")
    
    return model

def train_distillation():
    # 加载教师模型
    print("加载教师模型...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=device
    )
    teacher_model.eval()
    
    # 创建学生模型
    print("创建学生模型...")
    student_model = create_student_model()
    student_model = student_model.to(device)
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left"
    )
    
    # 确保tokenizer有必要的特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 准备数据集
    print("准备数据集...")
    train_dataset = QADataset("api_distill_data.json", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-6)  # 降低学习率
    
    # 学习率调度器
    num_epochs = 10  # 增加epoch数
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    # 训练循环
    print("开始训练...")
    student_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            # 将数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 获取教师模型的输出
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                teacher_logits = teacher_outputs.logits
            
            # 获取学生模型的输出
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            student_logits = student_outputs.logits
            
            # 计算蒸馏损失
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                alpha=0.5,
                temperature=1.0
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
        
        # 每个epoch保存一次模型
        print(f"保存第 {epoch + 1} 轮模型...")
        save_path = f"./api-student-epoch-{epoch + 1}"
        student_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    
    # 保存最终模型
    print("保存最终模型...")
    final_save_path = "./api-student-final"
    student_model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print("蒸馏完成！")

if __name__ == "__main__":
    train_distillation() 