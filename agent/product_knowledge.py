"""
商品知识库生成器

基于已有商品数据库的信息，自动生成售前客服常用的FAQ知识库：
- 商品基本信息QA
- 商品对比QA
- 选购建议QA
- 价格/库存/规格相关QA

生成的QA对会存入独立的向量库collection（product_faq），并保留与商品ID的关联关系。
"""

import sqlite3
import logging
import json
from typing import List, Dict, Any, Tuple


# 15个种子商品的结构化信息（与test_server_sse.py中的seed_data一致）
SEED_PRODUCTS = [
    {"id": "P001", "name": "iPhone 15 Pro Max", "category": "手机", "price": 9999.0, "brand": "Apple", "stock": 50,
     "description": "Apple最新旗舰手机，A17 Pro芯片，钛金属边框，4800万像素主摄",
     "highlights": ["A17 Pro芯片", "钛金属边框", "4800万像素主摄", "USB-C接口", "5倍光学变焦"]},
    {"id": "P002", "name": "华为Mate 60 Pro", "category": "手机", "price": 6999.0, "brand": "华为", "stock": 30,
     "description": "华为旗舰手机，麒麟9000S芯片，支持卫星通信，昆仑玻璃面板",
     "highlights": ["麒麟9000S芯片", "卫星通信", "昆仑玻璃", "鸿蒙系统", "XMAGE影像"]},
    {"id": "P003", "name": "小米14 Ultra", "category": "手机", "price": 5999.0, "brand": "小米", "stock": 80,
     "description": "小米影像旗舰，徕卡Summilux镜头，骁龙8 Gen3处理器",
     "highlights": ["徕卡Summilux镜头", "骁龙8 Gen3", "小米澎湃OS", "光影猎人900传感器", "75W无线快充"]},
    {"id": "P004", "name": "MacBook Pro 16寸", "category": "笔记本电脑", "price": 19999.0, "brand": "Apple", "stock": 20,
     "description": "M3 Max芯片，专业级笔记本，Liquid Retina XDR显示屏，续航22小时",
     "highlights": ["M3 Max芯片", "Liquid Retina XDR", "22小时续航", "专业级性能", "MagSafe充电"]},
    {"id": "P005", "name": "联想ThinkPad X1 Carbon", "category": "笔记本电脑", "price": 12999.0, "brand": "联想", "stock": 40,
     "description": "商务轻薄本，14英寸2.8K OLED屏，碳纤维机身，通过MIL-STD军标认证",
     "highlights": ["2.8K OLED屏", "碳纤维机身", "军标认证", "指纹+红外解锁", "Intel Evo平台"]},
    {"id": "P006", "name": "华为MateBook X Pro", "category": "笔记本电脑", "price": 11999.0, "brand": "华为", "stock": 25,
     "description": "华为旗舰轻薄本，3.1K触控屏，超级终端多设备协同",
     "highlights": ["3.1K触控屏", "超级终端", "多设备协同", "金属一体成型", "6扬声器"]},
    {"id": "P007", "name": "索尼WH-1000XM5", "category": "耳机", "price": 2499.0, "brand": "索尼", "stock": 100,
     "description": "旗舰降噪头戴式耳机，30mm驱动单元，自动降噪优化，30小时续航",
     "highlights": ["旗舰降噪", "30mm驱动单元", "30小时续航", "多点连接", "自适应降噪"]},
    {"id": "P008", "name": "AirPods Pro 2", "category": "耳机", "price": 1899.0, "brand": "Apple", "stock": 200,
     "description": "主动降噪无线耳机，H2芯片，自适应通透模式，USB-C充电",
     "highlights": ["H2芯片", "自适应通透", "个性化空间音频", "USB-C充电", "IP54防尘防水"]},
    {"id": "P009", "name": "iPad Air 5", "category": "平板电脑", "price": 4799.0, "brand": "Apple", "stock": 60,
     "description": "M1芯片平板电脑，10.9英寸Liquid Retina屏，支持Apple Pencil 2代",
     "highlights": ["M1芯片", "10.9英寸", "Apple Pencil 2代", "Center Stage", "5G可选"]},
    {"id": "P010", "name": "华为MatePad Pro 13.2", "category": "平板电脑", "price": 5699.0, "brand": "华为", "stock": 35,
     "description": "华为旗舰平板，13.2英寸OLED柔性屏，星闪连接，支持手写笔",
     "highlights": ["13.2英寸OLED", "星闪连接", "鸿蒙系统", "M-Pencil 3代", "PC级应用"]},
    {"id": "P011", "name": "戴森V15吸尘器", "category": "家电", "price": 4990.0, "brand": "戴森", "stock": 45,
     "description": "激光探测无绳吸尘器，可视化灰尘检测，240AW强劲吸力，60分钟续航",
     "highlights": ["激光探测", "灰尘可视化", "240AW吸力", "60分钟续航", "整机HEPA过滤"]},
    {"id": "P012", "name": "海尔冰箱BCD-500", "category": "家电", "price": 3299.0, "brand": "海尔", "stock": 15,
     "description": "500升对开门冰箱，风冷无霜，DEO净味，一级能效",
     "highlights": ["500升大容量", "风冷无霜", "DEO净味", "一级能效", "变频压缩机"]},
    {"id": "P013", "name": "Nike Air Max 270", "category": "运动鞋", "price": 899.0, "brand": "Nike", "stock": 150,
     "description": "经典气垫运动鞋，270度可视Air气垫，网面透气鞋面，轻量缓震",
     "highlights": ["270度Air气垫", "网面透气", "轻量缓震", "经典配色", "日常休闲跑步"]},
    {"id": "P014", "name": "Adidas Ultraboost", "category": "运动鞋", "price": 1299.0, "brand": "Adidas", "stock": 120,
     "description": "Boost缓震跑步鞋，Primeknit编织鞋面，Continental马牌橡胶外底",
     "highlights": ["Boost缓震", "Primeknit编织", "马牌橡胶外底", "专业跑步", "回弹性能"]},
    {"id": "P015", "name": "Apple Watch Ultra 2", "category": "智能手表", "price": 6499.0, "brand": "Apple", "stock": 30,
     "description": "户外探险智能手表，钛金属表壳，双频GPS精准定位，100米防水，36小时续航",
     "highlights": ["钛金属表壳", "双频GPS", "100米防水", "36小时续航", "S9芯片"]},
]


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
