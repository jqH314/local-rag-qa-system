"""
RAG 核心引擎
整合文档加载、分块、向量化、检索、问答全流程
"""
import os
from typing import List, Dict, Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document_loader import DocumentLoader
from core.vector_store import VectorStore
from core.llm import LLMClient
from utils.text_splitter import TextSplitter


class RAGEngine:
    """RAG问答引擎"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 4,
        vector_db_path: str = ""
    ):
        """
        初始化RAG引擎
        
        Args:
            chunk_size: 文本分块大小
            chunk_overlap: 分块重叠大小
            top_k: 检索返回的文档数量
            vector_db_path: 向量数据库本地保存路径
        """
        self.text_splitter = TextSplitter(chunk_size, chunk_overlap)
        self.vector_store = VectorStore()
        self.llm = LLMClient()
        self.top_k = top_k
        self.vector_db_path = vector_db_path
        
        # 如果有本地向量库，尝试加载
        if vector_db_path and os.path.exists(vector_db_path):
            self.vector_store.load_local(vector_db_path)
    
    def add_file(self, file_path: str) -> int:
        """
        添加单个文档到知识库
        
        Args:
            file_path: 文件路径
            
        Returns:
            添加的文档块数量
        """
        doc = DocumentLoader.load_file(file_path)
        chunks = self.text_splitter.split_text(doc['content'], doc['filename'])
        count = self.vector_store.add_documents(chunks)
        
        # 自动保存
        if self.vector_db_path:
            self.vector_store.save_local(self.vector_db_path)
        
        return count
    
    def add_files(self, file_paths: List[str]) -> int:
        """批量添加文档"""
        total = 0
        for path in file_paths:
            total += self.add_file(path)
        return total
    
    def add_directory(self, dir_path: str) -> int:
        """添加整个目录下的文档"""
        docs = DocumentLoader.load_directory(dir_path)
        chunks = self.text_splitter.split_documents(docs)
        count = self.vector_store.add_documents(chunks)
        
        if self.vector_db_path:
            self.vector_store.save_local(self.vector_db_path)
        
        return count
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        仅检索相关文档，不生成回答
        
        Args:
            query: 查询问题
            top_k: 返回数量，默认用初始化值
            
        Returns:
            相关文档列表
        """
        k = top_k or self.top_k
        return self.vector_store.similarity_search(query, k)
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict:
        """
        RAG问答：检索 + 生成回答
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            
        Returns:
            包含 answer、context、sources 的结果字典
        """
        # 1. 检索相关文档
        k = top_k or self.top_k
        context_docs = self.search(question, k)
        
        if not context_docs:
            return {
                'answer': '知识库为空，请先添加文档。',
                'context': [],
                'sources': []
            }
        
        # 2. 如果配置了LLM，生成回答
        if self.llm.is_configured():
            answer = self.llm.generate_rag_answer(question, context_docs)
        else:
            answer = "⚠️ 未配置大模型API，仅展示检索结果。请在设置中配置API密钥。"
        
        # 3. 提取来源
        sources = list(set([doc.get('source', '未知') for doc in context_docs]))
        
        return {
            'answer': answer,
            'context': context_docs,
            'sources': sources
        }
    
    def configure_llm(self, api_key: str, base_url: str, model: str):
        """配置大模型API"""
        self.llm = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
    
    def clear_knowledge_base(self):
        """清空知识库"""
        self.vector_store.clear()
        if self.vector_db_path and os.path.exists(self.vector_db_path):
            # 删除本地文件
            import shutil
            shutil.rmtree(self.vector_db_path, ignore_errors=True)
    
    @property
    def doc_count(self) -> int:
        """知识库中文档块数量"""
        return self.vector_store.document_count
