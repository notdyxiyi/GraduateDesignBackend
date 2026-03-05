"""
舍弃

"""

import socket
import os
# 1. 【最关键】在导入任何库之前，先设置环境变量
# 这样无论 unstructured 何时导入 pytesseract，都会读取到这个环境变量
os.environ['TESSERACT_CMD'] = r'D:\Tesseract-OCR\tesseract.exe'
# 同时也设置 pytesseract 推荐的变量名（以防万一）
os.environ['Tesseract_CMD'] = r'D:\Tesseract-OCR\tesseract.exe'
print(f"✅ 已设置环境变量 TESSERACT_CMD: {os.environ['TESSERACT_CMD']}")
import pytesseract
# 指定我自己的 Tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

# # 【关键】强制 socket 只使用 IPv4，避免 IPv6 解析失败
# # 这能解决很多 "getaddrinfo failed" 但浏览器能访问的问题
# old_getaddrinfo = socket.getaddrinfo
#
# def new_getaddrinfo(*args, **kwargs):
#     # 过滤掉 IPv6 (AF_INET6)，只保留 IPv4 (AF_INET)
#     results = old_getaddrinfo(*args, **kwargs)
#     return [res for res in results if res[0] == socket.AF_INET]
#
# socket.getaddrinfo = new_getaddrinfo


# 设置 Hugging Face 镜像地址 (国内极速)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from unstructured.partition.pdf import partition_pdf
# 配置
PDF_FILE = "../files/1.天津职业技术师范大学天津市人民政府奖学金评审及管理办法.pdf"

# 检查文件是否存在
if not os.path.exists(PDF_FILE):
    print(f"❌ 错误：找不到文件 '{PDF_FILE}'")
    print("💡 请在当前目录放一个 PDF 文件，并重命名为 test.pdf")
    exit(1)

print(f"🚀 开始解析 {PDF_FILE} ...")
print("💡 提示：如果是第一次运行，正在下载布局分析模型，可能需要几分钟，请耐心等待...")

try:
    # strategy="hi_res" 是关键！它会使用深度学习模型识别布局并提取坐标
    # model_name="yolox" 是默认的高效模型
    elements = partition_pdf(
        filename=PDF_FILE,
        strategy="hi_res",
        infer_table_structure=True,  # 尝试识别表格结构
        languages=["chi_sim", "eng"],  # 指定语言：简体中文 + 英文
    )
    # ================= 新增开始：文本校正逻辑 =================
    import pdfplumber

    print("🔧 正在使用 pdfplumber 提取原生文本进行校正...")

    native_lines = []
    # 1. 用 pdfplumber 提取所有页面的原生文本行
    with pdfplumber.open(PDF_FILE) as pdf:
        for page in pdf.pages:
            # extract_text 可能会把整页混在一起，这里尝试按行提取
            # x_tolerance=3 允许字符间有微小空隙视为同一行
            lines = page.extract_text(x_tolerance=3, y_tolerance=3)
            if lines:
                # 按换行符分割，去除空行
                native_lines.extend([line.strip() for line in lines.split('\n') if line.strip()])

    # 2. 简单的对齐策略：假设 OCR 提取的顺序和原生文本顺序基本一致
    # 如果数量差异巨大，说明切分粒度不同，此简单策略可能失效
    if len(native_lines) >= len(elements):
        print(f"ℹ️ 检测到 {len(native_lines)} 行原生文本，{len(elements)} 个 OCR 元素。尝试逐行覆盖...")
        native_idx = 0

        for i, el in enumerate(elements):
            ocr_text = el.text.strip()
            if not ocr_text:
                continue

            # 寻找最匹配的原生文本
            # 策略：找长度接近且开头几个字相同的行
            best_match = None
            min_diff = float('inf')

            # 只在当前索引附近搜索，避免错位太远
            search_range = range(max(0, native_idx - 2), min(len(native_lines), native_idx + 5))

            for k in search_range:
                cand = native_lines[k]
                # 长度差异不能超过 50%
                if abs(len(cand) - len(ocr_text)) > max(5, len(ocr_text) * 0.5):
                    continue

                # 计算相似度：如果前 3 个字相同，或者包含关键的“第x条”
                match_score = 0
                if len(ocr_text) > 2 and ocr_text[:2] == cand[:2]:
                    match_score += 10
                if "第" in ocr_text and "第" in cand and "条" in ocr_text and "条" in cand:
                    match_score += 5

                if match_score > 0:
                    best_match = cand
                    native_idx = k + 1  # 移动指针
                    break

            # 如果找到了高置信度匹配，替换文本
            if best_match:
                el.text = best_match
            else:
                # 没找到完美匹配，至少试试直接取当前索引的原生文本（如果长度差不多）
                if native_idx < len(native_lines):
                    cand = native_lines[native_idx]
                    if 0.5 < len(cand) / len(ocr_text) < 1.5:
                        el.text = cand
                        native_idx += 1

        print("✅ 文本校正完成。")
    else:
        print(f"⚠️ 警告：原生文本行数 ({len(native_lines)}) 远少于 OCR 元素数 ({len(elements)})，跳过自动校正以防错位。")
    # ================= 新增结束 =================

    print(f"\n✅ 解析成功！共提取到 {len(elements)} 个元素。\n")

    # 打印前 5 个元素的详细信息
    for i, el in enumerate(elements[:5]):
        meta = el.metadata.to_dict()
        print(f"--- [元素 {i + 1}] ---")
        print(f"类型: {el.category}")
        print(f"文本预览: {el.text[:60]}{'...' if len(el.text) > 60 else ''}")
        print(f"页码: {meta.get('page_number', 'N/A')}")

        # 检查坐标信息 (这是高亮的核心)
        coords = meta.get('coordinates')
        if coords:
            points = coords.get('points')
            if points:
                # points 格式: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] (顺时针)
                # 通常 points[0] 是左上角，points[2] 是右下角
                top_left = points[0]
                bottom_right = points[2]
                print(f"📍 坐标 (左上): ({top_left[0]:.2f}, {top_left[1]:.2f})")
                print(f"📍 坐标 (右下): ({bottom_right[0]:.2f}, {bottom_right[1]:.2f})")
            else:
                print("⚠️ 警告：有坐标元数据但无具体点位")
        else:
            print("⚠️ 警告：未检测到坐标信息 (可能是纯文本策略或解析失败)")
        print("")  # 空行

    print("🎉 测试完成！如果看到了坐标，说明环境完全正常，可以进行下一步开发了。")

except Exception as e:
    print(f"\n❌ 发生错误: {e}")
    print("\n💡 排查建议:")
    if "pdfinfo" in str(e):
        print("-> Poppler 问题：检查 PATH 环境变量是否包含 poppler 的 bin 目录。")
    elif "tesseract" in str(e).lower():
        print("-> Tesseract 问题：检查 PATH 环境变量，或确认 languages 参数是否正确 (应为 chi_sim)。")
    elif "No module named 'unstructured_inference'" or "torch" in str(e):
        print("-> Python 包缺失：请运行 'pip install unstructured-inference torch'。")
    else:
        print("-> 其他错误，请截图发给我。")