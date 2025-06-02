# Qwen2.5 医疗诊断微调项目

这个项目使用 Qwen2.5-0.5B 模型进行医疗诊断的微调训练。模型可以根据患者的年龄、性别和症状进行初步诊断。

## 项目结构

- `finetune_qwen.py`: 模型微调训练脚本
- `test_model.py`: 模型测试脚本
- `reduced_medical_data.jsonl`: 训练数据集

## 环境要求

- Python 3.8+
- PyTorch
- Transformers
- Datasets

## 安装

1. 克隆仓库：
```bash
git clone [你的仓库地址]
```

2. 创建虚拟环境：
```bash
python -m venv qwen_env
source qwen_env/bin/activate  # Linux/Mac
# 或
qwen_env\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 训练模型：
```bash
python finetune_qwen.py
```

2. 测试模型：
```bash
python test_model.py
```

## 数据格式

训练数据格式为 JSONL，每行包含：
```json
{
    "age": 65,
    "gender": "男",
    "symptoms": ["胸闷", "气短", "心悸"],
    "diagnosis": "高血压"
}
```

## 注意事项

- 训练需要较大的计算资源，建议使用 GPU
- 模型仅用于辅助诊断，不能替代专业医生的诊断
- 请确保遵守相关法律法规和医疗伦理规范 