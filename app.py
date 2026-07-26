"""
本地知识库 RAG 问答系统 - Streamlit 主界面
功能：
1. 上传文档构建知识库
2. 基于知识库智能问答
3. 展示检索来源与相关片段
"""
import streamlit as st
import os
import sys
import tempfile

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.rag_engine import RAGEngine


# 页面配置
st.set_page_config(
    page_title="智能知识库问答系统",
    page_icon="📚",
    layout="wide"
)

# 初始化 RAG 引擎（缓存）
@st.cache_resource
def get_rag_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, 'data', 'vector_db')
    return RAGEngine(vector_db_path=db_path)


# 初始化会话状态
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'doc_count' not in st.session_state:
    st.session_state.doc_count = 0


rag = get_rag_engine()

# 标题
st.title("📚 智能知识库问答系统")
st.markdown("基于 RAG 检索增强生成技术，上传文档后即可智能问答")
st.divider()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统设置")
    
    # LLM 配置
    st.subheader("🤖 大模型配置")
    
    api_provider = st.selectbox(
        "API 服务商",
        ["DeepSeek", "硅基流动", "OpenAI", "自定义"],
        help="选择OpenAI兼容的API服务"
    )
    
    # 预设配置
    provider_configs = {
        "DeepSeek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
        "硅基流动": {"base_url": "https://api.siliconflow.cn/v1", "model": "Qwen/Qwen2-7B-Instruct"},
        "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-3.5-turbo"},
        "自定义": {"base_url": "", "model": ""}
    }
    
    config = provider_configs[api_provider]
    
    base_url = st.text_input("API Base URL", value=config['base_url'])
    model_name = st.text_input("模型名称", value=config['model'])
    api_key = st.text_input("API Key", type="password", placeholder="输入你的API密钥")
    
    if st.button("💾 保存配置", use_container_width=True):
        if api_key and base_url and model_name:
            rag.configure_llm(api_key, base_url, model_name)
            st.success("✅ 配置已保存！")
        else:
            st.warning("请填写完整的API配置")
    
    st.divider()
    
    # 知识库状态
    st.subheader("📊 知识库状态")
    st.metric("文档块数量", rag.doc_count)
    
    if st.button("🗑️ 清空知识库", use_container_width=True, type="secondary"):
        rag.clear_knowledge_base()
        st.session_state.chat_history = []
        st.success("知识库已清空")
        st.rerun()
    
    st.divider()
    
    # 检索参数
    st.subheader("🔍 检索参数")
    top_k = st.slider("检索文档数量", min_value=1, max_value=10, value=4)
    chunk_size = st.slider("分块大小", min_value=200, max_value=1000, value=500, step=100)

# 主区域 - 两列布局
col_upload, col_chat = st.columns([1, 2])

with col_upload:
    st.subheader("📁 上传文档")
    
    uploaded_files = st.file_uploader(
        "上传文档构建知识库（支持 TXT / PDF / DOCX）",
        type=['txt', 'pdf', 'docx'],
        accept_multiple_files=True,
        help="可一次上传多个文件，自动分块向量化"
    )
    
    if uploaded_files and st.button("➕ 添加到知识库", type="primary", use_container_width=True):
        with st.spinner("正在处理文档..."):
            added_count = 0
            for file in uploaded_files:
                # 保存到临时文件
                suffix = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                try:
                    count = rag.add_file(tmp_path)
                    added_count += count
                finally:
                    os.unlink(tmp_path)
            
            st.success(f"✅ 成功添加 {added_count} 个文档块到知识库")
            st.session_state.doc_count = rag.doc_count
            st.rerun()
    
    st.info(f"💡 当前知识库：{rag.doc_count} 个文档块")
    
    # 示例文档提示
    with st.expander("📖 没有文档？试试示例"):
        st.markdown("""
        可以把项目目录下 `data/documents/` 里的示例文档上传测试。
        
        也可以自己创建一个 TXT 文件，粘贴一些文字内容上传。
        """)

with col_chat:
    st.subheader("💬 智能问答")
    
    # 显示对话历史
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.write(msg['content'])
            if msg['role'] == 'assistant' and msg.get('sources'):
                with st.expander("📄 参考来源"):
                    for i, src in enumerate(msg['sources'], 1):
                        st.markdown(f"{i}. {src}")
    
    # 输入框
    question = st.chat_input("输入你的问题，基于知识库内容回答...")
    
    if question:
        # 添加用户消息
        st.session_state.chat_history.append({'role': 'user', 'content': question})
        with st.chat_message('user'):
            st.write(question)
        
        # 生成回答
        with st.chat_message('assistant'):
            with st.spinner("正在检索并生成回答..."):
                result = rag.query(question, top_k=top_k)
            
            st.write(result['answer'])
            
            # 展示参考文档
            if result['context']:
                with st.expander("📄 相关文档片段"):
                    for i, doc in enumerate(result['context'], 1):
                        score = round(doc.get('score', 0) * 100, 1)
                        st.markdown(f"**片段{i}** （相关度：{score}%）- 来源：{doc.get('source', '未知')}")
                        st.markdown(f"> {doc['content'][:300]}..." if len(doc['content']) > 300 else f"> {doc['content']}")
                        st.divider()
        
        # 添加助手消息到历史
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': result['answer'],
            'sources': result['sources']
        })

# 页脚
st.divider()
st.caption("💡 本地知识库 RAG 问答系统 | FAISS向量库 + BERT语义检索 + 大模型生成 | 数据全部本地存储")
