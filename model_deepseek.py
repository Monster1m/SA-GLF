import csv
import time
import requests
import pandas as pd
import os
import sys
import logging
from tqdm import tqdm
from typing import Dict, List

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("node_embedding.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('NodeEmbedding')


class NodeEmbeddingGenerator:
    def __init__(self,
                 ollama_url: str = "http://localhost:11434",
                 model_name: str = "deepseek-r1:1.5b",
                 timeout: int = 90):
        """
        Ollama 节点嵌入生成器

        参数:
        ollama_url -- Ollama API 地址 (默认 "http://localhost:11434")
        model_name -- 模型名称 (默认 "deepseek-r1:1.5b")
        timeout -- API 请求超时时间 (默认 90 秒)
        """
        self.base_url = ollama_url.rstrip('/')
        self.model = model_name
        self.timeout = timeout
        self.dim = None
        logger.info(f"初始化嵌入生成器，模型: {self.model}")

    def check_api_available(self) -> bool:
        """检查 Ollama API 是否可用"""
        try:
            # 使用 /api/tags 端点检查 API 状态
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # 检查模型是否在可用模型列表中
            models = [model["name"] for model in data.get("models", [])]
            if self.model not in models:
                logger.error(f"模型 {self.model} 不可用。可用模型: {models}")
                return False

            logger.info(f"API 可用，模型 {self.model} 已存在")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"API 连接失败: {str(e)}")
            return False

    def get_embedding_dimension(self) -> int:
        """获取模型嵌入维度"""
        if self.dim:
            return self.dim

        try:
            # 发送测试请求
            payload = {
                "model": self.model,
                "prompt": "test"
            }
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # 验证响应格式
            if "embedding" not in data:
                logger.error(f"无效的API响应: {response.text}")
                raise ValueError(f"响应中缺少'embedding'字段")

            embedding = data["embedding"]
            self.dim = len(embedding)
            logger.info(f"嵌入维度: {self.dim}")
            return self.dim

        except Exception as e:
            logger.error(f"获取嵌入维度失败: {str(e)}")
            # 回退到默认维度
            self.dim = 1536 # deepseek-r1:1.5b 的标准维度
            logger.info(f"使用默认维度: {self.dim}")
            return self.dim

    def get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入"""
        payload = {
            "model": self.model,
            "prompt": text
        }

        for attempt in range(3):  # 最多尝试3次
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()

                if "embedding" not in data:
                    raise ValueError("响应中缺少'embedding'字段")

                return data["embedding"]

            except Exception as e:
                logger.warning(f"嵌入请求失败 ({attempt + 1}/3): {str(e)}")
                if attempt < 2:
                    time.sleep(1)  # 短暂等待后重试
                else:
                    logger.error(f"多次尝试后失败: {text[:50]}...")
                    return []

    def process_csv_file(self, input_path: str, output_path: str) -> bool:
        """处理单个CSV文件，将每行数据发送给DeepSeek生成嵌入"""
        logger.info(f"处理文件: {input_path}")

        if not os.path.exists(input_path):
            logger.error(f"输入文件不存在: {input_path}")
            return False

        # 检查API可用性
        if not self.check_api_available():
            raise ConnectionError("Ollama API 不可用")

        # 确保维度已知
        if self.dim is None:
            self.get_embedding_dimension()

        try:
            # 读取CSV文件
            df = pd.read_csv(input_path)

            # 验证列
            if "node_id" not in df.columns:
                logger.error("CSV缺少必要的列: 'node_id'")
                return False

            # 获取所有特征列（排除node_id）
            feature_columns = [col for col in df.columns if col != "node_id"]
            logger.info(f"检测到 {len(feature_columns)} 个特征列")

            # 准备嵌入字典
            embeddings = {}
            total_rows = len(df)

            # 进度条
            progress_bar = tqdm(total=total_rows, desc="生成嵌入")

            # 处理每一行
            for _, row in df.iterrows():
                node_id = row["node_id"]

                # 将特征值转换为字符串（用空格分隔）
                feature_values = [str(row[col]) for col in feature_columns]
                feature_text = " ".join(feature_values)

                try:
                    embedding = self.get_text_embedding(feature_text)

                    if embedding:
                        if len(embedding) != self.dim:
                            # 修正维度问题
                            if len(embedding) > self.dim:
                                embedding = embedding[:self.dim]
                            else:
                                embedding = embedding + [0.0] * (self.dim - len(embedding))
                        embeddings[node_id] = embedding
                    else:
                        # 创建空嵌入
                        embeddings[node_id] = [0.0] * self.dim

                    progress_bar.update(1)

                except Exception as e:
                    logger.error(f"节点 {node_id} 处理失败: {str(e)}")
                    embeddings[node_id] = [0.0] * self.dim
                    progress_bar.update(1)

            progress_bar.close()
            logger.info(f"成功处理 {len(embeddings)}/{total_rows} 个节点的嵌入")

            # 保存结果
            return self.save_to_csv(embeddings, output_path)

        except Exception as e:
            logger.error(f"处理文件失败: {str(e)}")
            return False

    def save_to_csv(self, embeddings: Dict[int, List[float]], output_path: str) -> bool:
        """将嵌入向量保存为CSV文件"""
        if not embeddings:
            raise ValueError("无嵌入数据可保存")

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        # 确保文件扩展名
        if not output_path.lower().endswith('.csv'):
            output_path += '.csv'

        # 准备数据
        dim = self.dim
        columns = ["node_id"] + [f"dim_{i}" for i in range(dim)]

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

            # 写入数据行
            for node_id, vector in embeddings.items():
                writer.writerow([node_id] + vector)

        logger.info(f"成功保存 {len(embeddings)} 个嵌入到: {output_path}")
        return True


def generate_node_embeddings(
        input_csv_1: str,
        input_csv_2: str,
        output_csv_1: str,
        output_csv_2: str,
        model_name: str = "deepseek-r1:1.5b"
):
    """为两个网络生成嵌入向量"""
    logger.info("=" * 60)
    logger.info(f"🏁 启动节点嵌入生成 - 模型: {model_name}")
    start_time = time.time()

    # 创建嵌入生成器
    try:
        embedder = NodeEmbeddingGenerator(model_name=model_name)
    except Exception as e:
        logger.error(f"创建嵌入生成器失败: {str(e)}")
        return False

    # 处理第一个文件
    logger.info("\n" + "=" * 20 + " 处理网络A " + "=" * 20)
    if not embedder.process_csv_file(input_csv_1, output_csv_1):
        logger.error("🛑 网络A处理失败")
        return False

    # 处理第二个文件
    logger.info("\n" + "=" * 20 + " 处理网络B " + "=" * 20)
    if not embedder.process_csv_file(input_csv_2, output_csv_2):
        logger.error("🛑 网络B处理失败")
        return False

    # 完成统计
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ 节点嵌入生成成功完成! 用时: {elapsed:.2f}秒")
    logger.info(f"• 网络A嵌入: {output_csv_1}")
    logger.info(f"• 网络B嵌入: {output_csv_2}")

    # 提供验证命令
    logger.info("\nAPI 测试命令:")
    logger.info(f'curl -X POST {embedder.base_url}/api/embeddings -d \'{{"model":"{model_name}", "prompt":"test"}}\'')

    return True


# 使用示例
def model_deepseek(dec_a, des_b, emb_a, emb_b):
    """入口函数"""
    # 示例路径
    DESCR_A = dec_a
    DESCR_B = des_b
    EMBED_A = emb_a
    EMBED_B = emb_b

    # 生成嵌入
    success = generate_node_embeddings(
        input_csv_1=DESCR_A,
        input_csv_2=DESCR_B,
        output_csv_1=EMBED_A,
        output_csv_2=EMBED_B,
        model_name="deepseek-r1:1.5b"
    )

    if success:
        logger.info("🎉 所有处理成功完成！")
        sys.exit(0)
    else:
        logger.error("❌ 处理失败")
        sys.exit(1)