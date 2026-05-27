#!/usr/bin/env python3
"""
文档添加工具

用于将文本文件添加到SmartAgent的知识库中
"""

import argparse
import os
import sys
from agent.agent import SmartAgent


def add_single_file(agent: SmartAgent, file_path: str, topic: str = None):
    """添加单个文件到知识库"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)

    print(f"📄 读取文件: {file_path}")

    try:
        # 使用 RAG 引擎的 DocumentLoader 加载文件
        content = agent.rag_engine.doc_loader.load_single_file(file_path)

        print(f"📝 文件大小: {len(content)} 字符")

        # 准备元数据
        metadata = {
            "source": os.path.basename(file_path),
            "file_path": file_path,
            "file_type": os.path.splitext(file_path)[1].lower()
        }

        if topic:
            metadata["topic"] = topic

        # 添加到知识库
        print("🔄 正在添加到知识库...")
        agent.add_documents_to_knowledge_base([content], [metadata])

        print("✅ 文档添加成功！")

        # 显示知识库信息
        info = agent.get_knowledge_base_info()
        print(f"\n📊 知识库统计:")
        print(f"  总文档块数: {info['document_count']}")
    except Exception as e:
        print(f"❌ 处理文件失败: {e}")
        sys.exit(1)


def add_directory(agent: SmartAgent, dir_path: str, extensions: list = None, topic: str = None):
    """添加目录中的所有文档文件"""
    if not os.path.isdir(dir_path):
        print(f"❌ 目录不存在: {dir_path}")
        sys.exit(1)

    if extensions is None:
        extensions = ['.txt', '.md', '.rst', '.pdf', '.docx', '.pptx']

    print(f"📁 扫描目录: {dir_path}")
    print(f"🔍 支持的扩展名: {', '.join(extensions)}")

    # 查找所有匹配的文件
    files = []
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))

    if not files:
        print(f"⚠️  未找到匹配的文件")
        return

    print(f"\n找到 {len(files)} 个文件:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {os.path.basename(f)}")

    # 读取所有文件
    documents = []
    metadatas = []

    print("\n📖 正在读取文件...")
    for file_path in files:
        try:
            # 使用 RAG 引擎的 DocumentLoader 加载文件
            content = agent.rag_engine.doc_loader.load_single_file(file_path)
            documents.append(content)

            metadata = {
                "source": os.path.basename(file_path),
                "file_path": file_path,
                "file_type": os.path.splitext(file_path)[1].lower()
            }
            if topic:
                metadata["topic"] = topic

            metadatas.append(metadata)
            print(f"  ✓ {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  ✗ {os.path.basename(file_path)}: {e}")

    if not documents:
        print("❌ 没有成功读取任何文件")
        return

    # 添加到知识库
    print(f"\n🔄 正在添加 {len(documents)} 个文档到知识库...")
    agent.add_documents_to_knowledge_base(documents, metadatas)

    print("✅ 所有文档添加成功！")

    # 显示知识库信息
    info = agent.get_knowledge_base_info()
    print(f"\n📊 知识库统计:")
    print(f"  总文档块数: {info['document_count']}")


def interactive_add(agent: SmartAgent):
    """交互式添加文档"""
    print("\n" + "=" * 60)
    print("📝 交互式文档添加")
    print("=" * 60)
    print("\n请输入文档内容（输入 'EOF' 结束）:")

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == 'EOF':
                break
            lines.append(line)
        except EOFError:
            break

    if not lines:
        print("⚠️  未输入任何内容")
        return

    content = '\n'.join(lines)

    topic = input("\n主题（可选，直接回车跳过）: ").strip()
    source = input("来源（可选，直接回车跳过）: ").strip()

    metadata = {}
    if topic:
        metadata["topic"] = topic
    if source:
        metadata["source"] = source
    else:
        metadata["source"] = "手动输入"

    print("\n🔄 正在添加到知识库...")
    agent.add_documents_to_knowledge_base([content], [metadata])

    print("✅ 文档添加成功！")

    # 显示知识库信息
    info = agent.get_knowledge_base_info()
    print(f"\n📊 知识库统计:")
    print(f"  总文档块数: {info['document_count']}")


def main():
    parser = argparse.ArgumentParser(
        description="添加文档到SmartAgent知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 添加单个文件
  python add_documents.py --file document.txt --topic "AI"
  
  # 添加目录中的所有文本文件
  python add_documents.py --directory ./docs --topic "技术文档"
  
  # 交互式添加
  python add_documents.py --interactive
  
  # 添加目录，指定文件类型
  python add_documents.py --directory ./docs --extensions .txt .md
        """
    )

    parser.add_argument(
        '--file', '-f',
        type=str,
        help='要添加的文件路径'
    )

    parser.add_argument(
        '--directory', '-d',
        type=str,
        help='要添加的目录路径（递归处理）'
    )

    parser.add_argument(
        '--topic', '-t',
        type=str,
        help='文档主题/分类'
    )

    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        help='目录模式下要处理的文件扩展名（默认: .txt .md .rst）'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='交互式输入文档内容'
    )

    args = parser.parse_args()

    # 检查参数
    if not (args.file or args.directory or args.interactive):
        parser.print_help()
        sys.exit(1)

    # 初始化Agent
    print("🚀 正在初始化SmartAgent...")
    try:
        agent = SmartAgent()
        print("✅ SmartAgent 初始化完成！\n")
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        print("请检查Ollama是否正在运行\n")
        sys.exit(1)

    # 执行相应操作
    try:
        if args.interactive:
            interactive_add(agent)
        elif args.file:
            add_single_file(agent, args.file, args.topic)
        elif args.directory:
            extensions = args.extensions if args.extensions else ['.txt', '.md', '.rst']
            # 确保扩展名以点开头
            extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
            add_directory(agent, args.directory, extensions, args.topic)
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
