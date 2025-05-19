# config.py

# 模型配置参数
MODEL_NAME = "/root/autodl-tmp/qwen/Qwen2-7B-Instruct"
DEVICE = "cuda"
TORCH_DTYPE = "auto"
DEVICE_MAP = "auto"

# 生成参数
TEMPERATURE = 0.95
TOP_P = 0.95
TOP_K = 50
MAX_NEW_TOKENS = 512

# 输入配置
PROMPT = "你好，今天天气怎么样？"