#!/usr/bin/env bash
python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --save_every 1 --max_tokens 1000

python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --save_every 1 --max_tokens 1000

python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --save_every 1 --max_tokens 1000

python generate_responses.py --model Qwen/QwQ-32B --save_every 1 --max_tokens 1000 --engine vllm

python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --save_every 1 --max_tokens 1000 --engine vllm

# python generate_responses.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --save_every 1 --max_tokens 1000 --engine vllm