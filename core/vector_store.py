"""
向量数据库模块
基于 FAISS 实现轻量级本地向量存储
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss


class VectorStore:
    """FAISS向量数据库"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化向量数据库
        
        Args:
            model_name: 向量化模型名称
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.documents: List[Dict] = []  # 存储文档元数据
    
    @property
    def model(self) -> SentenceTransformer:
        """懒加载模型"""
        if self._model is None:
            print(f"加载向量化模型: {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
            print("模型加载完成")
        return self._model
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """批量文本向量化"""
        if not texts:
            return np.array([])
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings).astype('float32')
    
    def add_documents(self, chunks: List[Dict]) -> int:
        """
        批量添加文档块到向量库
        
        Args:
            chunks: 文档块列表，每个包含 content、source、chunk_index
            
        Returns:
            添加的文档数量
        """
        if not chunks:
            return 0
        
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self._embed_texts(texts)
        
        # 初始化或更新FAISS索引
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            # 使用内积（因为向量已经归一化，内积等价于余弦相似度）
            self.index = faiss.IndexFlatIP(dimension)
        
        self.index.add(embeddings)
        self.documents.extend(chunks)
        
        return len(chunks)
    
    def similarity_search(self, query: str, top_k: int = 4) -> List[Dict]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            top_k: 返回最相关的k个结果
            
        Returns:
            相关文档块列表，带相似度分数
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        query_vec = self._embed_texts([query])
        scores, indices = self.index.search(query_vec, min(top_k, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results
    
    def save_local(self, save_dir: str):
        """
        保存向量库到本地
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(save_dir, 'faiss.index'))
        
        # 保存文档元数据
        with open(os.path.join(save_dir, 'documents.pkl'), 'wb') as f:
            pickle.dump(self.documents, f)
    
    def load_local(self, save_dir: str) -> bool:
        """
        从本地加载向量库
        
        Args:
            save_dir: 保存目录
            
        Returns:
            是否加载成功
        """
        index_path = os.path.join(save_dir, 'faiss.index')
        docs_path = os.path.join(save_dir, 'documents.pkl')
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            return False
        
        try:
            self.index = faiss.read_index(index_path)
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            return True
        except Exception as e:
            print(f"加载向量库失败: {e}")
            return False
    
    def clear(self):
        """清空向量库"""
        self.index = None
        self.documents = []
    
    @property
    def document_count(self) -> int:
        """文档块数量"""
        return len(self.documents)
