# Qwen2.5 Medical Diagnosis Fine-tuning Project

This project uses the Qwen2.5-0.5B model for fine-tuning medical diagnosis. The model can perform preliminary diagnosis based on patient's age, gender, and symptoms.

## Project Structure

- `finetune_qwen.py`: Model fine-tuning training script
- `test_model.py`: Model testing script
- `reduced_medical_data.jsonl`: Training dataset

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets

## Installation

1. Clone the repository:
```bash
git clone [your repository URL]
```

2. Create a virtual environment:
```bash
python -m venv qwen_env
source qwen_env/bin/activate  # Linux/Mac
# or
qwen_env\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python finetune_qwen.py
```

2. Test the model:
```bash
python test_model.py
```

## Data Format

The training data format is JSONL, with each line containing:
```json
{
    "age": 65,
    "gender": "male",
    "symptoms": ["chest tightness", "shortness of breath", "palpitations"],
    "diagnosis": "hypertension"
}
```

## Important Notes

- Training requires significant computational resources, GPU is recommended
- The model is only for auxiliary diagnosis and cannot replace professional medical diagnosis
- Please ensure compliance with relevant laws, regulations, and medical ethics guidelines 