# main.py

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import config

file_path = 'extracted_qa.json'
with open(file_path, 'r') as file:
    data = json.load(file)
question = data[0]['QUESTION']
print(question)

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    torch_dtype=config.TORCH_DTYPE,
    device_map=config.DEVICE_MAP
)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

# 准备输入
prompt = config.PROMPT
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(config.DEVICE)

# 生成回答
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=config.MAX_NEW_TOKENS,
    temperature=config.TEMPERATURE,
    top_p=config.TOP_P,
    top_k=config.TOP_K
)

# 处理生成的ID
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码生成的回答
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)