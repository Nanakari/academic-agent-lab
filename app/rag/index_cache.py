import hashlib  # 导入 hashlib，用于根据文件信息生成缓存 key
import json  # 导入 json，用于保存和读取 chunk 文本
from dataclasses import dataclass  # 导入 dataclass，用于定义缓存索引数据结构
from pathlib import Path  # 导入 Path，用于处理文件路径
from typing import List  # 导入 List，用于类型标注

import numpy as np  # 导入 numpy，用于保存和加载 embedding 向量矩阵

RAG_CACHE_VERSION = "v5_pdf_main_text_only"  # 定义缓存版本，确保参考文献截断规则不会复用旧索引



@dataclass  # 使用 dataclass 简化数据类定义
class CachedRagIndex:  # 定义缓存后的 RAG 索引结构
    chunks: List[str]  # 保存文本切分后的 chunk 列表
    vectors: np.ndarray  # 保存每个 chunk 对应的 embedding 向量
    cache_key: str  # 保存当前索引对应的缓存 key
    loaded_from_cache: bool  # 标记当前索引是否来自缓存


class RagIndexCache:  # 定义 RAG 索引缓存管理类
    def __init__(self) -> None:  # 初始化缓存管理器
        self.root_dir = Path(__file__).resolve().parents[2]  # 获取项目根目录 academic-agent-lab
        self.cache_dir = self.root_dir / "outputs" / "rag_cache"  # 设置缓存目录为 outputs/rag_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # 如果缓存目录不存在，就自动创建

    def build_cache_key(  # 定义生成缓存 key 的方法
        self,  # 当前缓存管理对象
        file_path: Path,  # 被检索的论文文件路径
        chunk_size: int,  # 当前使用的 chunk_size
        overlap: int,  # 当前使用的 overlap
        model_name: str,  # 当前使用的 embedding 模型名称
    ) -> str:  # 返回字符串形式的缓存 key
        stat = file_path.stat()  # 获取文件状态信息，例如大小和修改时间

        cache_info = {  # 构造参与缓存 key 计算的信息
            "path": str(file_path.resolve()),  # 保存文件绝对路径，避免不同文件同名冲突
            "rag_cache_version": RAG_CACHE_VERSION,  # 加入清洗版本以自动隔离旧缓存
            "size": stat.st_size,  # 保存文件大小，文件内容变化时通常会变化
            "mtime_ns": stat.st_mtime_ns,  # 保存文件纳秒级修改时间，用于判断文件是否更新
            "chunk_size": chunk_size,  # 保存切分大小，切分策略变了就不能复用旧缓存
            "overlap": overlap,  # 保存重叠长度，切分策略变了就不能复用旧缓存
            "model_name": model_name,  # 保存 embedding 模型名，模型变了向量也不能复用
        }  # 缓存信息构造结束

        text = json.dumps(cache_info, ensure_ascii=False, sort_keys=True)  # 把缓存信息稳定地序列化为字符串

        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]  # 计算 sha256，并截取前 24 位作为缓存 key

    def get_cache_paths(self, cache_key: str) -> tuple[Path, Path, Path]:  # 根据缓存 key 获取三个缓存文件路径
        chunk_path = self.cache_dir / f"{cache_key}_chunks.json"  # chunks 缓存文件路径
        vector_path = self.cache_dir / f"{cache_key}_vectors.npy"  # vectors 缓存文件路径
        meta_path = self.cache_dir / f"{cache_key}_meta.json"  # meta 缓存文件路径
        return chunk_path, vector_path, meta_path  # 返回三个缓存文件路径

    def load(self, cache_key: str) -> CachedRagIndex | None:  # 根据缓存 key 尝试加载缓存索引
        chunk_path, vector_path, meta_path = self.get_cache_paths(cache_key)  # 获取缓存文件路径

        if not chunk_path.exists():  # 如果 chunk 缓存不存在
            return None  # 返回 None，表示缓存未命中

        if not vector_path.exists():  # 如果向量缓存不存在
            return None  # 返回 None，表示缓存未命中

        if not meta_path.exists():  # 如果元信息缓存不存在
            return None  # 返回 None，表示缓存未命中

        try:  # 捕获缓存读取失败的问题
            chunks = json.loads(chunk_path.read_text(encoding="utf-8"))  # 从 json 文件加载 chunk 列表
            vectors = np.load(vector_path)  # 从 npy 文件加载 embedding 向量矩阵

            return CachedRagIndex(  # 返回缓存索引对象
                chunks=chunks,  # 保存加载到的 chunks
                vectors=vectors,  # 保存加载到的 vectors
                cache_key=cache_key,  # 保存缓存 key
                loaded_from_cache=True,  # 标记这是从缓存加载的
            )  # CachedRagIndex 创建结束

        except Exception:  # 如果缓存文件损坏或读取失败
            return None  # 返回 None，让上层重新构建索引

    def save(self, cache_key: str, chunks: List[str], vectors: np.ndarray) -> None:  # 保存 RAG 索引到缓存
        chunk_path, vector_path, meta_path = self.get_cache_paths(cache_key)  # 获取缓存文件路径

        chunk_path.write_text(  # 写入 chunks json 文件
            json.dumps(chunks, ensure_ascii=False, indent=2),  # 将 chunks 转换为格式化 json 字符串
            encoding="utf-8",  # 使用 UTF-8 编码写入
        )  # chunks 写入结束

        np.save(vector_path, vectors)  # 将 embedding 向量矩阵保存为 npy 文件

        meta = {  # 构造元信息
            "cache_key": cache_key,  # 保存缓存 key
            "chunk_count": len(chunks),  # 保存 chunk 数量
            "vector_shape": list(vectors.shape),  # 保存向量矩阵形状
        }  # 元信息构造结束

        meta_path.write_text(  # 写入 meta json 文件
            json.dumps(meta, ensure_ascii=False, indent=2),  # 将 meta 转换为格式化 json
            encoding="utf-8",  # 使用 UTF-8 编码写入
        )  # meta 写入结束