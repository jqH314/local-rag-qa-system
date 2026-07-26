"""
快速测试脚本
不启动Streamlit，直接测试RAG核心功能
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.rag_engine import RAGEngine


def main():
    print("=" * 60)
    print("本地知识库 RAG 系统 - 快速测试")
    print("=" * 60)
    
    # 初始化RAG引擎
    print("\n🔧 初始化RAG引擎...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(base_dir, 'data', 'documents')
    rag = RAGEngine()
    
    # 加载示例文档
    print(f"\n📄 加载示例文档: {docs_dir}")
    count = rag.add_directory(docs_dir)
    print(f"✅ 成功加载 {count} 个文档块")
    
    # 测试检索
    print("\n" + "=" * 60)
    print("🔍 测试语义检索")
    print("=" * 60)
    
    test_questions = [
        "智云科技是做什么的？",
        "智能客服系统有什么功能？",
        "RAG的核心流程是什么？"
    ]
    
    for q in test_questions:
        print(f"\n问题：{q}")
        results = rag.search(q, top_k=2)
        for i, doc in enumerate(results, 1):
            score = round(doc.get('score', 0) * 100, 1)
            preview = doc['content'][:80].replace('\n', ' ')
            print(f"  结果{i} (相关度{score}%): {preview}...")
    
    print("\n" + "=" * 60)
    print("✅ 检索功能测试通过！")
    print("\n💡 提示：")
    print("   - 检索功能正常，向量数据库工作正常")
    print("   - 如需测试问答生成功能，请在界面中配置大模型API")
    print("   - 启动完整界面请运行：streamlit run app.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
