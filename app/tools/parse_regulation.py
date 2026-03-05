"""
舍弃

"""
import re
import os
# 设置 Hugging Face 镜像地址 (国内极速)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from typing import List, Dict, Any, Optional
from unstructured.partition.pdf import partition_pdf

# --- 配置区域 ---
PDF_FILE = "../files/1.天津职业技术师范大学天津市人民政府奖学金评审及管理办法.pdf"  # 你的文件名
OUTPUT_JSON = "regulations_structured.json"

# --- 正则表达式定义 (针对中文规章制度) ---
# 匹配 "第一章", "第二章", ...
PATTERN_CHAPTER = re.compile(r"^第[零一二三四五六七八九十百]+章")
# 匹配 "第一条", "第二条", ... (可能包含 "第 1 条")
PATTERN_ARTICLE = re.compile(r"^第[零一二三四五六七八九十百0-9]+条")
# 匹配 "(一)", "(二)", "1.", "（1）" 等款项
PATTERN_CLAUSE = re.compile(r"^([\(（]?[零一二三四五六七八九十百0-9]+[\)）\.]|[(（][0-9]+[)）])")


def classify_element_type(text: str) -> str:
    """根据文本内容判断层级类型"""
    if not text:
        return "unknown"

    # 去除首尾空格
    clean_text = text.strip()

    if PATTERN_CHAPTER.match(clean_text):
        return "chapter"
    elif PATTERN_ARTICLE.match(clean_text):
        return "article"
    elif PATTERN_CLAUSE.match(clean_text):
        return "clause"
    else:
        return "content"  # 普通正文内容


def extract_coordinates(element) -> Dict[str, Any]:
    """从 element 中提取标准化的坐标信息"""
    meta = element.metadata.to_dict()
    coords = meta.get('coordinates', {})
    points = coords.get('points', [])

    if not points:
        return None

    # points 通常是 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] (顺时针)
    # 我们提取左上角 (top_left) 和 右下角 (bottom_right) 方便前端计算
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    return {
        "page": meta.get('page_number', 1),
        "x_min": min(x_coords),
        "y_min": min(y_coords),
        "x_max": max(x_coords),
        "y_max": max(y_coords),
        "points": points  # 保留原始多边形点集，方便前端做精确多边形高亮
    }


def parse_regulations(pdf_path: str) -> List[Dict[str, Any]]:
    print(f"🚀 开始解析规章制度：{pdf_path}")

    # 1. 使用 hi_res 策略获取带坐标的原始元素
    elements = partition_pdf(
        filename=pdf_path,
        strategy="fast",
        infer_table_structure=True,
        languages=["chi_sim", "eng"]
    )

    structured_data = []

    # 2. 遍历并重组逻辑
    # 注意：unstructured 返回的顺序通常已经是阅读顺序，我们直接利用这个顺序进行层级推断
    current_chapter = None
    current_article = None

    for i, el in enumerate(elements):
        text = el.text.strip()
        if not text:
            continue

        el_type = classify_element_type(text)
        coords = extract_coordinates(el)

        # 构建基础节点
        node = {
            "id": f"elem_{i}",
            "type": el_type,  # chapter, article, clause, content
            "text": text,
            "category": el.category,  # 原始类别 (Title, Narrative, etc.)
            "coordinates": coords,  # 核心高亮数据
            "children": []  # 用于嵌套结构 (可选，看前端需求)
        }

        # 3. 简单的层级状态机 (也可以直接平铺，靠 type 字段区分)
        # 这里我们采用“平铺 + 上下文引用”的方式，方便前端渲染列表
        # 如果需要严格的树形结构，可以在这里做嵌套逻辑

        if el_type == "chapter":
            current_chapter = text
            current_article = None  # 新章节开始，重置条
            node["context"] = {"chapter": text, "article": None}

        elif el_type == "article":
            current_article = text
            node["context"] = {"chapter": current_chapter, "article": text}

        elif el_type == "clause":
            node["context"] = {"chapter": current_chapter, "article": current_article}

        else:  # content
            node["context"] = {"chapter": current_chapter, "article": current_article}

        structured_data.append(node)

    return structured_data


if __name__ == "__main__":
    try:
        result = parse_regulations(PDF_FILE)

        # 保存结果
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 解析完成！共提取 {len(result)} 个结构化片段。")
        print(f"💾 数据已保存至：{OUTPUT_JSON}")

        # 打印前 3 条预览
        print("\n--- 数据预览 ---")
        for item in result[:3]:
            print(
                f"[{item['type']}] {item['text'][:30]}... | 页码:{item['coordinates']['page'] if item['coordinates'] else 'N/A'}")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback

        traceback.print_exc()