import pymupdf4llm
import os
import glob
import re
import time
from typing import Optional, List, Tuple

# --- 配置常量 (可根据需要修改) ---
DEFAULT_OUTPUT_DIR = "../markdown"


def _ensure_dir(directory: str):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def _clean_markdown_text(text: str) -> str:
    """
    清洗 Markdown 文本：
    1. 去除所有空格
    2. 压缩多余换行符 (3个及以上变为2个)
    """
    # 去除所有空格
    cleaned = text.replace(" ", "")
    # 压缩多余换行
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned


def convert_single_pdf_to_md(
        pdf_path: str,
        output_path: Optional[str] = None,
        remove_spaces: bool = True,
        compress_newlines: bool = True
) -> Tuple[bool, str, str]:
    """
    将单个 PDF 文件转换为 Markdown。

    参数:
        pdf_path: PDF 文件的绝对或相对路径。
        output_path: (可选) 输出 .md 文件的路径。如果为 None，则不保存文件，仅返回字符串。
                     如果提供了路径但没指定文件名，会自动生成同名 .md 文件。
        remove_spaces: 是否去除所有空格 (默认 True)。
        compress_newlines: 是否压缩多余换行符 (默认 True)。

    返回:
        Tuple[success: bool, message: str, content: str]
        - success: 转换是否成功
        - message: 成功或错误信息
        - content: 转换后的 Markdown 内容 (即使失败也返回已处理的部分或空字符串)
    """
    if not os.path.exists(pdf_path):
        return False, f"文件不存在: {pdf_path}", ""

    try:
        # 1. 执行转换
        md_text = pymupdf4llm.to_markdown(pdf_path)

        # 2. 清洗文本
        if remove_spaces:
            md_text = md_text.replace(" ", "")
        if compress_newlines:
            md_text = re.sub(r'\n{3,}', '\n\n', md_text)

        # 3. 保存文件 (如果指定了输出路径)
        if output_path:
            # 如果 output_path 是目录，则自动生成文件名
            if os.path.isdir(output_path):
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                final_path = os.path.join(output_path, f"{base_name}.md")
            else:
                # 确保输出目录存在
                out_dir = os.path.dirname(output_path)
                if out_dir:
                    _ensure_dir(out_dir)
                final_path = output_path

            with open(final_path, "w", encoding="utf-8") as f:
                f.write(md_text)

            return True, f"成功保存至: {final_path}", md_text

        return True, "转换成功 (未保存文件)", md_text

    except Exception as e:
        return False, f"转换失败: {str(e)}", ""


def batch_convert_pdfs_to_md(
        input_dir: str,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        remove_spaces: bool = True,
        compress_newlines: bool = True
) -> Tuple[int, int, List[str]]:
    """
    批量将指定目录下的所有 PDF 转换为 Markdown。

    参数:
        input_dir: 包含 PDF 文件的输入目录路径。
        output_dir: 输出 Markdown 文件的目录路径 (自动创建)。
        remove_spaces: 是否去除所有空格。
        compress_newlines: 是否压缩多余换行符。

    返回:
        Tuple[success_count, fail_count, error_logs]
        - success_count: 成功数量
        - fail_count: 失败数量
        - error_logs: 失败文件的错误信息列表
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    _ensure_dir(output_dir)

    # 获取所有 PDF 文件
    pdf_files = []
    for ext in ["*.pdf", "*.PDF"]:
        pdf_files.extend(glob.glob(os.path.join(input_dir, ext)))
    pdf_files = list(set(pdf_files))  # 去重

    if not pdf_files:
        return 0, 0, [f"在 '{input_dir}' 中未找到任何 PDF 文件"]

    print(f"🔍 发现 {len(pdf_files)} 个 PDF 文件，开始批量转换...\n")

    success_count = 0
    fail_count = 0
    error_logs = []

    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        print(f"[{i}/{len(pdf_files)}] 正在处理: {filename} ...")

        # 构造输出路径 (同目录下同名.md)
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, f"{base_name}.md")

        success, msg, _ = convert_single_pdf_to_md(
            pdf_path=pdf_path,
            output_path=out_path,
            remove_spaces=remove_spaces,
            compress_newlines=compress_newlines
        )

        if success:
            print(f"   ✅ {msg}")
            success_count += 1
        else:
            print(f"   ❌ {msg}")
            error_logs.append(f"{filename}: {msg}")
            fail_count += 1

    print("-" * 60)
    print(f"🎉 批量处理完成！成功: {success_count}, 失败: {fail_count}")
    if error_logs:
        print("⚠️ 错误详情:")
        for log in error_logs:
            print(f"   - {log}")

    return success_count, fail_count, error_logs


# --- 示例用法 (当直接运行此脚本时) ---
if __name__ == "__main__":
    # 配置路径
    INPUT_FOLDER = "../files"
    OUTPUT_FOLDER = "../markdown"

    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ 错误：输入目录 '{INPUT_FOLDER}' 不存在。")
        exit(1)

    start_time = time.time()

    # 调用批量处理函数
    s_count, f_count, logs = batch_convert_pdfs_to_md(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER
    )

    end_time = time.time()
    print(f"\n⏱️ 总耗时: {end_time - start_time:.2f} 秒")

    # --- 如果需要单独测试单个文件，取消下面注释 ---
    # single_pdf = "../files/1.天津职业技术师范大学天津市人民政府奖学金评审及管理办法.pdf"
    # success, msg, content = convert_single_pdf_to_md(single_pdf, output_dir=OUTPUT_FOLDER)
    # print(f"单文件测试: {msg}")
    # if success:
    #     print(f"前200字符预览:\n{content[:200]}")