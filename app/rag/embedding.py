from typing import List  # 导入 List，用于标注字符串列表和向量列表

import numpy as np  # 导入 numpy，用于保存和计算向量
from sentence_transformers import SentenceTransformer  # 导入 SentenceTransformer，用于加载本地 embedding 模型


class LocalEmbeddingModel:  # 定义本地 embedding 模型封装类
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:  # 初始化方法，默认使用多语言 E5 小模型
        self.model_name = model_name  # 保存模型名称，方便调试和后续配置
        self.model = SentenceTransformer(model_name)  # 加载本地 embedding 模型，首次运行会下载模型

    def encode_query(self, query: str) -> np.ndarray:  # 定义编码查询文本的方法
        text = f"query: {query}"  # E5 检索模型要求查询前加 query 前缀
        vector = self.model.encode(  # 调用模型生成 query 向量
            text,  # 传入带前缀的查询文本
            normalize_embeddings=True,  # 对向量做归一化，方便直接用点积表示相似度
        )  # 编码结束
        return np.asarray(vector, dtype=np.float32)  # 转成 numpy float32 向量并返回

    def encode_passages(self, passages: List[str]) -> np.ndarray:  # 定义编码文档片段的方法
        texts = [f"passage: {passage}" for passage in passages]  # E5 检索模型要求文档片段前加 passage 前缀
        vectors = self.model.encode(  # 调用模型批量生成 passage 向量
            texts,  # 传入带前缀的文档片段列表
            normalize_embeddings=True,  # 对所有向量做归一化
        )  # 编码结束
        return np.asarray(vectors, dtype=np.float32)  # 转成 numpy float32 矩阵并返回