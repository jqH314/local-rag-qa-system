"""
大模型调用模块
支持 OpenAI 兼容 API 接口，可对接各类大模型服务
"""
from typing import Optional, List, Dict
from openai import OpenAI


class LLMClient:
    """大模型客户端"""
    
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat"
    ):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥
            base_url: API地址（OpenAI兼容格式）
            model: 模型名称
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client: Optional[OpenAI] = None
    
    @property
    def client(self) -> OpenAI:
        """获取客户端实例"""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client
    
    def is_configured(self) -> bool:
        """检查是否已配置API"""
        return bool(self.api_key and self.base_url and self.model)
    
    def chat(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        """
        单轮对话
        
        Args:
            prompt: 用户提问
            system_prompt: 系统提示词
            temperature: 温度参数（0-1，越小越确定）
            
        Returns:
            模型回复文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False
        )
        
        return response.choices[0].message.content
    
    def generate_rag_answer(
        self,
        question: str,
        context_docs: List[Dict],
        temperature: float = 0.3
    ) -> str:
        """
        RAG问答：基于检索到的文档生成答案
        
        Args:
            question: 用户问题
            context_docs: 检索到的相关文档列表
            temperature: 温度参数
            
        Returns:
            生成的答案
        """
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.get('source', '未知文档')
            content = doc['content']
            context_parts.append(f"【文档{i} - 来源：{source}】\n{content}")
        
        context_text = "\n\n".join(context_parts)
        
        system_prompt = """你是一个专业的文档问答助手。请根据提供的参考文档内容回答用户的问题。
要求：
1. 只基于参考文档中的内容回答，不要编造信息
2. 如果参考文档中没有相关内容，请回答"根据现有文档无法回答该问题"
3. 回答要清晰、准确、有条理
4. 可以适当引用原文，但不要大段复制"""
        
        user_prompt = f"""参考文档：
{context_text}

用户问题：{question}

请根据参考文档回答问题："""
        
        return self.chat(user_prompt, system_prompt, temperature)
