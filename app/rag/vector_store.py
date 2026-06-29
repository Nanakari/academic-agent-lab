from dataclasses import dataclass  # 导入 dataclass，用于快速定义检索结果结构
from typing import List  # 导入 List，用于类型标注

import numpy as np  # 导入 numpy，用于向量矩阵计算

from app.rag.embedding import LocalEmbeddingModel  # 导入本地 embedding 模型封装类


@dataclass  # 使用 dataclass 简化检索结果类
class SearchResult:  # 定义检索结果结构
    chunk_id: int  # 保存片段编号
    text: str  # 保存片段文本
    score: float  # 保存语义相似度得分


class EmbeddingVectorStore:  # 定义 embedding 向量库
    def __init__(self, embedding_model: LocalEmbeddingModel | None = None) -> None:  # 初始化向量库
        self.embedding_model = embedding_model or LocalEmbeddingModel()  # 如果外部没有传模型，就创建默认本地模型
        self.chunks: List[str] = []  # 保存原始文本片段
        self.vectors: np.ndarray | None = None  # 保存片段向量矩阵

    def add_texts(self, chunks: List[str]) -> None:  # 添加文本片段到向量库
        self.chunks = chunks  # 保存原始 chunk 文本
        self.vectors = self.embedding_model.encode_passages(chunks)  # 使用本地 embedding 模型编码所有 chunk

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:  # 根据用户问题检索相关片段
        if not self.chunks:  # 如果向量库中没有文本片段
            return []  # 返回空结果

        if self.vectors is None:  # 如果向量矩阵还没有建立
            return []  # 返回空结果

        query_vector = self.embedding_model.encode_query(query)  # 使用本地 embedding 模型编码用户问题

        scores = self.vectors @ query_vector  # 因为向量已归一化，所以点积就是余弦相似度

        top_k = min(top_k, len(self.chunks))  # 防止 top_k 超过 chunk 数量

        top_indices = np.argsort(scores)[::-1][:top_k]  # 按得分从高到低取前 top_k 个索引

        results = []  # 创建检索结果列表

        for index in top_indices:  # 遍历得分最高的片段索引
            results.append(  # 添加一个检索结果
                SearchResult(  # 创建 SearchResult 对象
                    chunk_id=int(index) + 1,  # chunk 编号从 1 开始，方便用户阅读
                    text=self.chunks[int(index)],  # 保存对应 chunk 文本
                    score=float(scores[int(index)]),  # 保存相似度得分
                )  # SearchResult 创建结束
            )  # append 结束

        return results  # 返回检索结果列表