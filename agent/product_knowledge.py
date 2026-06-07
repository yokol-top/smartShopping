"""
商品知识库生成器

基于 init/product_descriptions.json 中的商品信息，自动生成售前客服FAQ知识库。
同时将 init/product_descriptions.json 中的详细描述素材一并导入向量库。

生成的文档会存入向量库，并保留与商品ID的关联关系。
"""

import os
import sqlite3
import logging
import json
from typing import List, Dict, Any, Tuple

# 商品描述素材文件路径
_INIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'init')
_DESCRIPTIONS_FILE = os.path.join(_INIT_DIR, 'product_descriptions.json')
_DB_PATH = os.path.join(os.path.dirname(_INIT_DIR), 'data', 'db', 'app.db')


def _load_products_from_json() -> List[Dict[str, Any]]:
    """从 init/product_descriptions.json 加载商品信息"""
    if not os.path.exists(_DESCRIPTIONS_FILE):
        logging.warning(f"商品描述文件不存在: {_DESCRIPTIONS_FILE}")
        return []
    with open(_DESCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def _get_stock_from_db() -> Dict[str, int]:
    """从 SQLite 数据库读取各商品的实时库存，返回 {product_id: stock} 映射"""
    if not os.path.exists(_DB_PATH):
        logging.warning(f"[product_knowledge] 数据库不存在，库存将显示为0: {_DB_PATH}")
        return {}
    try:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        rows = conn.execute("SELECT id, stock FROM products").fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except Exception as e:
        logging.warning(f"[product_knowledge] 查询数据库库存失败，库存将显示为0: {e}")
        return {}


def _get_seed_products() -> List[Dict[str, Any]]:
    """获取种子商品列表（兼容旧接口）"""
    raw = _load_products_from_json()
    stock_map = _get_stock_from_db()
    products = []
    for p in raw:
        if p.get('product_id', '').startswith('P'):
            pid = p['product_id']
            products.append({
                'id': pid,
                'name': p['product_name'],
                'category': p['category'],
                'brand': p.get('brand', ''),
                'price': p.get('price', 0),
                'stock': stock_map.get(pid, 0),
                'description': '',
            })
    return products


# 兼容旧代码引用
SEED_PRODUCTS = _get_seed_products()


def generate_product_faqs() -> List[Dict[str, Any]]:
    """基于商品信息生成FAQ知识条目

    Returns:
        每个条目包含: question, answer, product_id, product_name, category, faq_type
    """
    faqs = []

    for p in SEED_PRODUCTS:
        pid = p["id"]
        name = p["name"]
        cat = p["category"]
        price = p["price"]
        brand = p["brand"]
        stock = p["stock"]
        desc = p["description"]
        highlights = p.get("highlights", [])
        hl_str = "、".join(highlights) if highlights else desc

        # --- 1. 商品基本信息 ---
        faqs.append({
            "question": f"{name}有什么特点？",
            "answer": f"[{pid}] {name}是{brand}品牌的{cat}产品，售价¥{price}。主要特点：{hl_str}。{desc}",
            "product_id": pid, "product_name": name, "category": cat, "faq_type": "product_info"
        })
        faqs.append({
            "question": f"{name}多少钱？价格是多少？",
            "answer": f"[{pid}] {name}的售价是¥{price}，当前库存{stock}件。",
            "product_id": pid, "product_name": name, "category": cat, "faq_type": "price"
        })
        faqs.append({
            "question": f"{name}还有货吗？库存多少？",
            "answer": f"[{pid}] {name}当前库存{stock}件，{'库存充足' if stock > 20 else '库存有限，建议尽快购买'}。",
            "product_id": pid, "product_name": name, "category": cat, "faq_type": "stock"
        })

        # --- 2. 选购建议 ---
        if cat == "手机":
            if price >= 9000:
                faqs.append({
                    "question": f"{name}适合什么人群？",
                    "answer": f"[{pid}] {name}售价¥{price}，属于旗舰定位，适合追求极致性能和品牌体验的用户。{hl_str}，特别适合摄影爱好者和商务人士。",
                    "product_id": pid, "product_name": name, "category": cat, "faq_type": "recommendation"
                })
            elif price >= 6000:
                faqs.append({
                    "question": f"{name}适合什么人群？",
                    "answer": f"[{pid}] {name}售价¥{price}，属于高端旗舰，性价比出色。{hl_str}，适合注重性能和创新功能的用户。",
                    "product_id": pid, "product_name": name, "category": cat, "faq_type": "recommendation"
                })
            else:
                faqs.append({
                    "question": f"{name}适合什么人群？",
                    "answer": f"[{pid}] {name}售价¥{price}，在旗舰手机中价格最亲民。{hl_str}，适合预算有限但追求旗舰体验的用户。",
                    "product_id": pid, "product_name": name, "category": cat, "faq_type": "recommendation"
                })

        elif cat == "笔记本电脑":
            faqs.append({
                "question": f"{name}适合什么用途？",
                "answer": f"[{pid}] {name}售价¥{price}，{desc}。{'适合专业创作者（视频剪辑、3D渲染等）' if price > 15000 else '适合商务办公和轻度创作'}。核心卖点：{hl_str}。",
                "product_id": pid, "product_name": name, "category": cat, "faq_type": "recommendation"
            })

        elif cat == "耳机":
            faqs.append({
                "question": f"{name}降噪效果怎么样？",
                "answer": f"[{pid}] {name}是{brand}的{'旗舰头戴式' if price > 2000 else '真无线'}降噪耳机，售价¥{price}。{hl_str}。降噪效果在同价位中属于顶级水准。",
                "product_id": pid, "product_name": name, "category": cat, "faq_type": "feature_detail"
            })

    # --- 3. 同类商品对比 ---
    categories = {}
    for p in SEED_PRODUCTS:
        categories.setdefault(p["category"], []).append(p)

    for cat, products in categories.items():
        if len(products) >= 2:
            names = [p["name"] for p in products]
            prices = [f"[{p['id']}] {p['name']}(¥{p['price']})" for p in products]
            comparison = f"我们{cat}类目有{len(products)}款产品：{'、'.join(prices)}。"

            # 按价格排序
            sorted_prods = sorted(products, key=lambda x: x["price"])
            cheapest = sorted_prods[0]
            most_expensive = sorted_prods[-1]

            comparison += f"\n价格最亲民的是[{cheapest['id']}] {cheapest['name']}(¥{cheapest['price']})，"
            comparison += f"配置最高的是[{most_expensive['id']}] {most_expensive['name']}(¥{most_expensive['price']})。"

            for p in products:
                hl = "、".join(p.get("highlights", [])[:3])
                comparison += f"\n- [{p['id']}] {p['name']}：{hl}"

            faqs.append({
                "question": f"{cat}有哪些可以选？哪款{cat}好？{cat}推荐",
                "answer": comparison,
                "product_id": ",".join(p["id"] for p in products),
                "product_name": ",".join(names),
                "category": cat,
                "faq_type": "comparison"
            })

    # --- 4. 预算推荐 ---
    budget_ranges = [
        (0, 2000, "2000元以内"),
        (2000, 5000, "2000-5000元"),
        (5000, 10000, "5000-10000元"),
        (10000, 20000, "10000-20000元"),
    ]
    for low, high, label in budget_ranges:
        matching = [p for p in SEED_PRODUCTS if low <= p["price"] < high]
        if matching:
            items = [f"[{p['id']}] {p['name']}(¥{p['price']}, {p['category']})" for p in matching]
            faqs.append({
                "question": f"预算{label}买什么好？{label}有什么推荐？",
                "answer": f"在{label}预算范围内，我们有以下产品可选：{'、'.join(items)}。建议根据您的具体需求（如用途、品牌偏好）进一步筛选。",
                "product_id": ",".join(p["id"] for p in matching),
                "product_name": ",".join(p["name"] for p in matching),
                "category": "综合",
                "faq_type": "budget_recommendation"
            })

    # --- 5. 大学生/学生套装推荐 ---
    faqs.append({
        "question": "大学生买什么电子产品好？学生开学装备推荐",
        "answer": (
            "大学生电子装备推荐方案（按预算分档）：\n"
            "💻 笔记本电脑（必备）：[P006] 华为MateBook X Pro(¥11999)或[P005] 联想ThinkPad X1 Carbon(¥12999)\n"
            "📱 手机：[P003] 小米14 Ultra(¥5999)性价比最高，[P002] 华为Mate 60 Pro(¥6999)生态协同好\n"
            "🎧 耳机：[P008] AirPods Pro 2(¥1899)或[P007] 索尼WH-1000XM5(¥2499)\n"
            "📱 平板（可选）：[P009] iPad Air 5(¥4799)记笔记神器\n\n"
            "预算方案：\n"
            "- 1万左右：手机([P003] 小米14 Ultra ¥5999) + 耳机([P008] AirPods Pro 2 ¥1899) ≈ ¥7898\n"
            "- 1.5万左右：上述 + 平板([P009] iPad Air 5 ¥4799) ≈ ¥12697\n"
            "- 2万左右：笔记本([P006] MateBook X Pro ¥11999) + 手机([P003] 小米14 Ultra ¥5999) + 耳机([P008] AirPods Pro 2 ¥1899) ≈ ¥19897"
        ),
        "product_id": "P003,P006,P008,P009",
        "product_name": "小米14 Ultra,华为MateBook X Pro,AirPods Pro 2,iPad Air 5",
        "category": "综合",
        "faq_type": "bundle_recommendation"
    })

    return faqs


def build_product_knowledge_base(rag_engine, logger: logging.Logger = None,
                                  force_rebuild: bool = True):
    """将商品FAQ导入到RAG向量库

    仅在向量库为空时导入，避免重复启动时产生重复文档。
    当 force_rebuild=True 时，先清除旧数据再重新导入。

    Args:
        rag_engine: RAGEngine实例（使用其embeddings和vector_store）
        logger: 日志记录器
        force_rebuild: 是否强制重建（清除旧数据后重新导入）

    Returns:
        导入的文档数量（已存在时返回0）
    """
    logger = logger or logging.getLogger(__name__)

    # 强制重建：先清除旧的 product_faq 数据
    if force_rebuild:
        try:
            rag_engine.vector_store.delete(where={"source": "product_faq"})
            logger.info("已清除旧的商品FAQ数据，准备重新导入")
        except Exception as e:
            logger.warning(f"清除旧FAQ数据时出错: {e}，将尝试全量导入")

    # 检查向量库中是否已有 product_faq 数据，避免重复导入
    if not force_rebuild:
        try:
            existing_count = rag_engine.vector_store.get_collection_count()
            if existing_count > 0:
                # 尝试查询是否已有product_faq来源的文档
                test_results = rag_engine.vector_store.query(
                    query_texts=["商品FAQ"],
                    n_results=1,
                    where={"source": "product_faq"},
                )
                if test_results and test_results.get("ids") and test_results["ids"][0]:
                    logger.info(
                        f"商品FAQ已存在于向量库中（{existing_count} 条文档），跳过重复导入"
                    )
                    return 0
        except Exception as e:
            logger.debug(f"检查向量库状态时出错（首次运行属正常）: {e}")

    faqs = generate_product_faqs()
    logger.info(f"生成了 {len(faqs)} 条商品FAQ")

    # 准备文档和元数据
    documents = []
    metadatas = []

    for faq in faqs:
        # 将问题和答案合成为一个文档块，便于向量检索
        doc_text = f"问：{faq['question']}\n答：{faq['answer']}"
        documents.append(doc_text)

        metadata = {
            "source": "product_faq",
            "product_id": faq["product_id"],
            "product_name": faq["product_name"],
            "category": faq["category"],
            "faq_type": faq["faq_type"],
        }
        metadatas.append(metadata)

    # 添加到知识库
    rag_engine.add_documents(documents, metadatas)
    logger.info(f"商品FAQ已导入向量库，共 {len(documents)} 条")

    return len(documents)
