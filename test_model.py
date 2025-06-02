# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model():
    # 1. 加载模型和分词器
    print("正在加载模型和分词器...")
    model_path = "./qwen2.5-medical-finetune-final"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="mps"
    )
    print("模型加载完成")
    print("\n请输入病例信息（输入'q'退出）：")
    print("格式示例：年龄: 65，性别: 男，症状: 胸闷,气短,心悸")
    print("-" * 50)

    while True:
        # 获取用户输入
        user_input = input("\n请输入病例信息: ").strip()
        
        # 检查是否退出
        if user_input.lower() == 'q':
            print("程序已退出")
            break
            
        # 检查输入是否为空
        if not user_input:
            print("输入不能为空，请重新输入！")
            continue
            
        # 构建完整的提示词
        prompt = f"{user_input}。请给出诊断结果。"
        
        # 生成回答
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # 限制生成长度
            num_beams=5,  # 使用beam search
            early_stopping=True,  # 启用早停
            no_repeat_ngram_size=3,  # 避免重复
            do_sample=False  # 禁用采样，使用确定性生成
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取诊断结果
        try:
            # 分割并获取最后一个"答："后面的内容
            parts = response.split("答：")
            if len(parts) > 1:
                diagnosis = parts[-1].strip()
                # 移除.unbind后缀
                diagnosis = diagnosis.replace(".unbind", "")
                # 只保留第一个句号之前的内容
                diagnosis = diagnosis.split("。")[0]
                # 移除可能的多余内容
                diagnosis = diagnosis.split("性别:")[0].strip()
                # 移除可能的问答对
                diagnosis = diagnosis.split("提问:")[0].strip()
                diagnosis = diagnosis.split("回答:")[0].strip()
            else:
                diagnosis = "无法给出明确诊断"
        except:
            diagnosis = "无法给出明确诊断"
        
        # 打印结果
        print("\n诊断结果：")
        print(diagnosis)
        print("-" * 50)

if __name__ == "__main__":
    test_model() 