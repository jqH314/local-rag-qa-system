"""
文本分块工具
将长文档切分成适合向量化的小块
"""
import re
from typing import List, Dict


class TextSplitter:
    """文本分块器"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        初始化分块器
        
        Args:
            chunk_size: 每块的最大字符数
            chunk_overlap: 块之间的重叠字符数（保证上下文连续）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str, filename: str = "") -> List[Dict]:
        """
        将文本切分成多个块
        
        Args:
            text: 原始文本
            filename: 文件名，用于标记来源
            
        Returns:
            分块列表，每个块包含 content、source、chunk_index
        """
        if not text or not text.strip():
            return []
        
        # 先按段落粗略分割
        paragraphs = self._split_by_paragraph(text)
        
        # 合并成指定大小的块
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 如果当前块加上新段落还没超过限制，就加进去
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                # 当前块满了，保存
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # 开始新块，带上重叠部分
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    # 取上一块末尾的重叠部分
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n" + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"
        
        # 处理最后一块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # 包装成带元数据的格式
        result = []
        for idx, chunk in enumerate(chunks):
            result.append({
                'content': chunk,
                'source': filename,
                'chunk_index': idx
            })
        
        return result
    
    def _split_by_paragraph(self, text: str) -> List[str]:
        """按段落分割文本"""
        # 按空行分割段落
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        批量切分多个文档
        
        Args:
            documents: 文档列表，每个包含 filename 和 content
            
        Returns:
            所有分块的列表
        """
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc['content'], doc['filename'])
            all_chunks.extend(chunks)
        return all_chunks
