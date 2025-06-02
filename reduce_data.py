# -*- coding: utf-8 -*-
import json
import random

# 读取原始数据
with open('synthetic_medical_data.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 随机选择1000条数据
reduced_data = random.sample(data, 1000)

# 写入新文件
with open('reduced_medical_data.jsonl', 'w', encoding='utf-8') as f:
    for item in reduced_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"原始数据条数: {len(data)}")
print(f"减少后数据条数: {len(reduced_data)}") 