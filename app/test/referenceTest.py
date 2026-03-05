"""
引用溯源测试
"""
import fitz  # PyMuPDF
import json


def get_highlight_coords(pdf_path, keywords):
    """
    纯核心逻辑：读取 PDF -> 搜索关键词 -> 返回坐标列表
    不做任何文件校验，假设文件一定存在且可读
    """
    doc = fitz.open(pdf_path)
    results = []

    # 遍历每一页
    for page_num, page in enumerate(doc):
        # page_num 从 0 开始，返回给前端时 +1

        for keyword in keywords:
            # 搜索关键词，返回矩形列表 (Rect)
            rects = page.search_for(keyword)

            for rect in rects:
                results.append({
                    "page": page_num + 1,  # 前端页码 (1-based)
                    "keyword": keyword,  # 命中的词
                    "x0": rect.x0,  # 左上角 X
                    "y0": rect.y0,  # 左上角 Y (PDF坐标系: 左下角为原点)
                    "x1": rect.x1,  # 右下角 X
                    "y1": rect.y1,  # 右下角 Y
                    "width": rect.width,  # 宽度
                    "height": rect.height  # 高度
                })

    doc.close()

    return {
        "status": "success",
        "total_hits": len(results),
        "data": results
    }


# ================= 配置区域 =================

# 👇 在这里修改你的 PDF 文件名
PDF_FILE = "../files/1.天津职业技术师范大学天津市人民政府奖学金评审及管理办法.pdf"

# 👇 在这里修改你想高亮的关键词 (模拟 RAG 检索出的核心词)
SEARCH_KEYWORDS = ["奖学金", "第一条", "学生处"]

# ================= 执行 =================

if __name__ == "__main__":
    print(f"⚡ 正在处理：{PDF_FILE} ...")

    # 直接运行，不检查文件是否存在 (如果文件没了会直接报错，符合你的要求)
    response = get_highlight_coords(PDF_FILE, SEARCH_KEYWORDS)

    # 输出结果
    print(json.dumps(response, ensure_ascii=False, indent=2))

    if response["total_hits"] > 0:
        print(f"\n✅ 完成！共找到 {response['total_hits']} 处高亮。")
    else:
        print("\n⚠️ 未找到任何关键词，请检查关键词拼写或 PDF 内容。")