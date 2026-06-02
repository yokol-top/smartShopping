#!/usr/bin/env python3
"""
商品知识库导入脚本

从 product_descriptions.json 读取商品详细描述素材，导入到RAG向量库中。
每条素材保留与商品ID的关联，支持按商品/类目/类型检索。

用法:
    python -m init.import_product_kb                    # 默认导入（已有则跳过）
    python -m init.import_product_kb --force             # 强制重建（清除旧数据后重新导入）
    python -m init.import_product_kb --config ./config/settings.yaml
"""

import os
import sys
import json
import argparse
import logging

# 将项目根目录加入搜索路径
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 素材文件路径
INIT_DIR = os.path.dirname(os.path.abspath(__file__))
DESCRIPTIONS_FILE = os.path.join(INIT_DIR, 'product_descriptions.json')


def load_product_descriptions() -> list:
    """加载商品描述素材"""
    with open(DESCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_documents(products: list) -> tuple:
    """将商品描述素材转换为向量库文档格式

    Returns:
        (documents, metadatas) 两个列表
    """
    documents = []
    metadatas = []

    for product in products:
        pid = product['product_id']
        pname = product['product_name']
        category = product['category']
        brand = product.get('brand', '')
        price = product.get('price', 0)

        for doc in product.get('documents', []):
            doc_type = doc.get('type', 'detail')
            title = doc.get('title', '')
            content = doc.get('content', '')

            # 文档文本：标题 + 内容
            doc_text = f"# {title}\n\n{content}"

            # 元数据：保留与商品的关联
            metadata = {
                'source': 'product_description',
                'product_id': pid,
                'product_name': pname,
                'category': category,
                'brand': brand,
                'price': str(price),
                'doc_type': doc_type,
                'title': title,
            }

            documents.append(doc_text)
            metadatas.append(metadata)

    return documents, metadatas


def import_to_vector_db(
    config_path: str = None,
    force: bool = False,
):
    """导入商品描述到向量库

    Args:
        config_path: 配置文件路径
        force: 是否强制重建
    """
    from utils import ConfigLoader
    from rag.rag_engine import RAGEngine
    from utils import LLMClient

    # 加载配置
    config_path = config_path or os.path.join(_project_root, 'config', 'settings.yaml')
    config_loader = ConfigLoader(config_path)
    config = config_loader.config

    # 解析嵌入模型路径
    embedding_config = config.get('embedding', {}).copy()
    if embedding_config.get('provider') == 'local' and embedding_config.get('model'):
        embedding_config['model'] = config_loader.resolve_path(embedding_config['model'])

    config_with_paths = config.copy()
    config_with_paths['embedding'] = embedding_config

    # 初始化LLM客户端
    llm_client = LLMClient(
        api_key=config.get('llm', {}).get('api_key', 'EMPTY'),
        base_url=config.get('llm', {}).get('base_url', 'http://localhost:11434/v1'),
        model=config.get('llm', {}).get('model', 'qwen2.5:latest'),
        logger=logger,
    )

    # 初始化RAG引擎
    logger.info("初始化RAG引擎...")
    rag_engine = RAGEngine(config_with_paths, llm_client=llm_client, logger=logger)

    # 强制重建：先清除旧的商品描述数据
    if force:
        try:
            rag_engine.vector_store.delete(where={"source": "product_description"})
            logger.info("已清除旧的商品描述数据")
        except Exception as e:
            logger.warning(f"清除旧数据时出错: {e}")

    # 检查是否已有数据
    if not force:
        try:
            test_results = rag_engine.vector_store.query(
                query_texts=["商品描述"],
                n_results=1,
                where={"source": "product_description"},
            )
            if test_results and test_results.get("ids") and test_results["ids"][0]:
                logger.info("商品描述已存在于向量库中，跳过导入（使用 --force 强制重建）")
                return 0
        except Exception:
            pass  # 首次运行属正常

    # 加载素材
    logger.info(f"加载商品描述素材: {DESCRIPTIONS_FILE}")
    products = load_product_descriptions()
    logger.info(f"共 {len(products)} 个商品/指南")

    # 构建文档
    documents, metadatas = build_documents(products)
    logger.info(f"生成 {len(documents)} 条向量库文档")

    # 导入向量库
    logger.info("导入向量库...")
    rag_engine.add_documents(documents, metadatas)

    logger.info(f"✅ 商品描述导入完成，共 {len(documents)} 条文档")

    # 统计
    categories = set(m['category'] for m in metadatas)
    doc_types = {}
    for m in metadatas:
        doc_types[m['doc_type']] = doc_types.get(m['doc_type'], 0) + 1
    logger.info(f"   类目: {', '.join(categories)}")
    logger.info(f"   文档类型: {doc_types}")

    return len(documents)


def main():
    parser = argparse.ArgumentParser(description='导入商品描述到向量知识库')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (默认: ./config/settings.yaml)')
    parser.add_argument('--force', action='store_true',
                        help='强制重建（清除旧数据后重新导入）')
    args = parser.parse_args()

    count = import_to_vector_db(config_path=args.config, force=args.force)
    if count > 0:
        logger.info(f"导入 {count} 条商品描述文档")
    else:
        logger.info("未导入新文档")


if __name__ == '__main__':
    main()
