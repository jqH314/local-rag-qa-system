"""
文档加载模块
支持读取 TXT、PDF、DOCX 格式文档
"""
import os
from typing import List, Dict
from PyPDF2 import PdfReader
from docx import Document


class DocumentLoader:
    """文档加载器"""
    
    @staticmethod
    def load_txt(file_path: str) -> str:
        """读取TXT文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """读取PDF文件"""
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return '\n'.join(text_parts)
    
    @staticmethod
    def load_docx(file_path: str) -> str:
        """读取DOCX文件"""
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return '\n'.join(paragraphs)
    
    @classmethod
    def load_file(cls, file_path: str) -> Dict:
        """
        根据扩展名自动读取文件
        
        Returns:
            包含 filename 和 content 的字典
        """
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        
        if ext == '.txt':
            content = cls.load_txt(file_path)
        elif ext == '.pdf':
            content = cls.load_pdf(file_path)
        elif ext == '.docx':
            content = cls.load_docx(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        return {
            'filename': filename,
            'content': content,
            'file_path': file_path
        }
    
    @classmethod
    def load_directory(cls, dir_path: str) -> List[Dict]:
        """批量读取目录下所有支持的文档"""
        supported_ext = {'.txt', '.pdf', '.docx'}
        results = []
        
        if not os.path.isdir(dir_path):
            return results
        
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in supported_ext and os.path.isfile(file_path):
                try:
                    doc = cls.load_file(file_path)
                    results.append(doc)
                except Exception as e:
                    print(f"读取文件 {filename} 失败: {e}")
        
        return results
