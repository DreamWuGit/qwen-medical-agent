import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import OrderedDict
import numpy as np

def load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """加载模型"""
    print(f"加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=device
    )
    return model

def get_model_info(model):
    """获取模型信息"""
    config = model.config
    total_params = sum(p.numel() for p in model.parameters())
    
    info = {
        "总参数量": total_params,
        "层数": config.num_hidden_layers,
        "隐藏层维度": config.hidden_size,
        "注意力头数": config.num_attention_heads,
        "中间层维度": config.intermediate_size,
        "词表大小": config.vocab_size,
        "最大位置编码": config.max_position_embeddings
    }
    return info

def compare_models(teacher_path, student_path):
    """比较两个模型的差异"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    teacher_model = load_model(teacher_path, device)
    student_model = load_model(student_path, device)
    
    # 获取模型信息
    teacher_info = get_model_info(teacher_model)
    student_info = get_model_info(student_model)
    
    # 打印比较结果
    print("\n=== 模型比较 ===")
    print("\n教师模型 (deepseek-r1-distill-qwen-1.5b):")
    for key, value in teacher_info.items():
        print(f"{key}: {value:,}")
    
    print("\n学生模型 (deepseek1.5B-student):")
    for key, value in student_info.items():
        print(f"{key}: {value:,}")
    
    print("\n差异对比:")
    for key in teacher_info.keys():
        teacher_value = teacher_info[key]
        student_value = student_info[key]
        if isinstance(teacher_value, (int, float)):
            diff = ((student_value - teacher_value) / teacher_value) * 100
            print(f"{key}: {diff:+.2f}%")
    
    # 计算参数量比例
    params_ratio = (student_info["总参数量"] / teacher_info["总参数量"]) * 100
    print(f"\n参数量比例: 学生模型是教师模型的 {params_ratio:.2f}%")

if __name__ == "__main__":
    teacher_path = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
    student_path = "./deepseek1.5B-student-final"
    compare_models(teacher_path, student_path) 