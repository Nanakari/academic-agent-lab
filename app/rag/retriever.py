from pathlib import Path  # 导入 Path，用于处理文件路径
from typing import List  # 导入 List，用于类型标注

from app.rag.document_loader import load_document_text  # 导入统一文档读取函数，用于支持 txt、md 和文本型 PDF
from app.rag.index_cache import RagIndexCache  # 导入 RAG 索引缓存管理器
from app.rag.splitter import split_text  # 导入文本切分函数
from app.rag.vector_store import EmbeddingVectorStore, SearchResult  # 导入 embedding 向量库和检索结果结构


def retrieve_from_file(  # 定义从文件中检索相关片段的函数
    file_path: Path,  # 论文文件路径
    query: str,  # 用户查询
    top_k: int = 3,  # 返回片段数量
    chunk_size: int = 800,  # 每个 chunk 的最大字符数
    overlap: int = 100,  # chunk 之间的重叠字符数
) -> List[SearchResult]:  # 返回 SearchResult 列表
    cache = RagIndexCache()  # 创建 RAG 缓存管理器

    vector_store = EmbeddingVectorStore()  # 创建 embedding 向量库，同时加载本地 embedding 模型

    cache_key = cache.build_cache_key(  # 根据文件状态、切分参数和模型名称生成缓存 key
        file_path=file_path,  # 传入文件路径
        chunk_size=chunk_size,  # 传入 chunk_size
        overlap=overlap,  # 传入 overlap
        model_name=vector_store.embedding_model.model_name,  # 传入 embedding 模型名称
    )  # 缓存 key 生成结束

    cached_index = cache.load(cache_key)  # 尝试从缓存中加载已经构建好的 RAG 索引

    if cached_index is not None:  # 如果缓存命中
        vector_store.chunks = cached_index.chunks  # 直接使用缓存中的 chunks，避免重新读取和切分
        vector_store.vectors = cached_index.vectors  # 直接使用缓存中的 vectors，避免重新计算 chunk embedding
        return vector_store.search(query=query, top_k=top_k)  # 只对 query 编码并执行检索

    text = load_document_text(file_path)  # 缓存未命中时，通过统一文档读取层加载论文文本

    chunks = split_text(text=text, chunk_size=chunk_size, overlap=overlap)  # 缓存未命中时，切分论文文本

    if not chunks:  # 如果没有切出有效 chunks
        return []  # 返回空检索结果

    vector_store.add_texts(chunks)  # 缓存未命中时，计算所有 chunk 的 embedding

    if vector_store.vectors is not None:  # 如果向量矩阵成功生成
        cache.save(  # 保存 RAG 索引缓存
            cache_key=cache_key,  # 传入缓存 key
            chunks=vector_store.chunks,  # 保存 chunks
            vectors=vector_store.vectors,  # 保存向量矩阵
        )  # 缓存保存结束

    return vector_store.search(query=query, top_k=top_k)  # 执行检索并返回结果


def retrieve_from_text(  # 保留旧函数，方便已有代码或测试继续使用
    text: str,  # 论文文本
    query: str,  # 用户查询
    top_k: int = 3,  # 返回片段数量
    chunk_size: int = 800,  # 每个 chunk 的最大字符数
    overlap: int = 100,  # chunk 之间的重叠字符数
) -> List[SearchResult]:  # 返回 SearchResult 列表
    chunks = split_text(text=text, chunk_size=chunk_size, overlap=overlap)  # 把文本切分成 chunks

    if not chunks:  # 如果 chunks 为空
        return []  # 返回空列表

    vector_store = EmbeddingVectorStore()  # 创建 embedding 向量库

    vector_store.add_texts(chunks)  # 添加 chunks 并计算 embedding

    return vector_store.search(query=query, top_k=top_k)  # 执行语义检索并返回结果