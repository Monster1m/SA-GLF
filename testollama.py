import requests
import json
import time


def ask_deepseek(prompt, model="deepseek-r1:1.5b", max_retries=3, retry_delay=2, temperature=0.7, max_tokens=4096):
    """
    使用本地部署的 Ollama DeepSeek 模型进行问答

    参数:
        prompt: 问题或提示文本
        model: Ollama 模型名称 (默认为 "deepseek")
        max_retries: 最大重试次数 (默认3)
        retry_delay: 重试延迟秒数 (默认2)
        temperature: 生成温度 (0.0-1.0, 默认0.7)
        max_tokens: 最大生成token数 (默认4096)

    返回:
        模型生成的回答文本
    """
    # Ollama API 端点
    url = "http://localhost:11434/api/embeddings"

    # 请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 请求体
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)

    # 检查响应状态
    if response.status_code == 200:
        # 解析JSON响应
        response_data = response.json()

        # 检查响应是否完成
        if response_data.get("done", False):
            return response_data.get("response", "")
        else:
            raise ValueError("模型未完成响应")


# 使用示例
if __name__ == "__main__":
    # 测试问题
    question = "介绍图神经网络"

    # 获取回答
    answer = ask_deepseek(question)

    print(f"问题: {question}")
    print(f"回答: {answer}")