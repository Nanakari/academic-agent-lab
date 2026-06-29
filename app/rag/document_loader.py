import re  # 导入正则表达式模块，用于识别小数、章节标题和压缩空白
import unicodedata  # 导入 Unicode 分类模块，用于识别数学符号和标点符号
from pathlib import Path  # 导入 Path，用于接收和判断文档文件路径

import fitz  # 导入 PyMuPDF 的 fitz 模块，用于提取文本型 PDF 的页面文字


PDF_NO_TEXT_MESSAGE = "未能从 PDF 中提取到可用文本。这可能是扫描版 PDF，需要 OCR 才能识别文字。"  # 定义扫描版或无可用文字 PDF 的统一提示
ACADEMIC_HEADINGS = {"abstract", "introduction", "background", "related work", "method", "methods", "methodology", "approach", "framework", "experiment", "experiments", "results", "discussion", "conclusion", "conclusions", "references"}  # 定义需要保留的常见论文章节标题
MATH_AND_CHART_SYMBOLS = set("=±×÷∑∏√∞≈≠≤≥∫∆∂∇^_{}[]|<>→←↔⊕⊗∈∉∪∩′″")  # 定义常见数学和图表符号集合
AXIS_WORDS = {"actual", "predicted", "tested"}  # 定义常见坐标轴或图表残片词语


def is_noise_line(line: str) -> bool:  # 定义行级噪声判断函数，用于识别 PDF 图表、公式和坐标轴残片
    normalized_line = re.sub(r"\s+", " ", line).strip()  # 合并连续空白并清理行首行尾空白

    if not normalized_line:  # 判断当前行是否为空
        return True  # 空行本身不作为有效文本行保留

    heading_text = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s*", "", normalized_line, flags=re.IGNORECASE).lower()  # 移除章节编号后得到标题正文

    if heading_text in ACADEMIC_HEADINGS:  # 判断当前行是否为需要保留的常见论文章节标题
        return False  # 明确保留章节标题，避免短行规则误删语义锚点

    if normalized_line.lower() in AXIS_WORDS:  # 判断当前行是否只是单独的坐标轴标签词
        return True  # 过滤单独出现的 Actual、Predicted 或 tested 图表标签

    visible_characters = [character for character in normalized_line if not character.isspace()]  # 收集所有非空白字符用于比例统计
    visible_count = len(visible_characters)  # 计算当前行的可见字符数量

    if visible_count < 4:  # 判断当前行是否短到几乎不包含语义
        return True  # 过滤页码碎片、单个变量和极短坐标轴刻度

    letter_count = sum(character.isalpha() for character in visible_characters)  # 统计 Unicode 字母和中文字符数量
    digit_count = sum(character.isdigit() for character in visible_characters)  # 统计数字字符数量
    symbol_count = sum(unicodedata.category(character).startswith(("P", "S")) for character in visible_characters)  # 统计标点和符号字符数量
    math_symbol_count = sum(character in MATH_AND_CHART_SYMBOLS or unicodedata.category(character) == "Sm" for character in visible_characters)  # 统计数学和图表符号数量
    styled_math_letter_count = sum(0x1D400 <= ord(character) <= 0x1D7FF for character in visible_characters)  # 统计数学字母 Unicode 区段中的特殊变量字符
    variable_fragment_count = len(re.findall(r"(?<![A-Za-z])[A-Za-z][0-9′']*(?![A-Za-z])", normalized_line))  # 统计 P0、R′、T0 等短变量残片数量
    digit_ratio = digit_count / visible_count  # 计算数字字符占全部可见字符的比例
    symbol_ratio = symbol_count / visible_count  # 计算标点和符号占全部可见字符的比例
    math_symbol_ratio = math_symbol_count / visible_count  # 计算数学和图表符号占全部可见字符的比例
    decimal_count = len(re.findall(r"(?<!\w)[+-]?(?:\d+\.\d+|\.\d+)(?!\w)", normalized_line))  # 统计独立小数数量以识别坐标轴刻度
    word_set = {word.lower() for word in re.findall(r"[A-Za-z]+", normalized_line)}  # 提取小写英文词集合用于识别图表轴标签
    axis_word_count = len(word_set & AXIS_WORDS)  # 统计当前行命中的坐标轴特征词数量

    if digit_count >= 4 and digit_ratio > 0.35:  # 判断当前行是否由大量数字主导
        return True  # 过滤坐标轴刻度、数值表格行和数字序列

    if (styled_math_letter_count >= 1 or variable_fragment_count >= 3) and letter_count <= 8 and digit_count + symbol_count >= 3:  # 判断当前行是否主要由多个数学变量残片组成
        return True  # 过滤数学字母、下标变量和带撇号变量组成的残片行

    if letter_count <= 2 and symbol_count >= 4 and symbol_ratio > 0.35:  # 判断当前行是否字母极少但符号密集
        return True  # 过滤公式碎片、图例符号和无语义标记

    if (math_symbol_count >= 3 or letter_count <= 3 and math_symbol_count >= 2) and math_symbol_ratio > 0.18 and letter_count < 20:  # 判断当前行是否包含密集数学符号且自然语言不足
        return True  # 过滤公式、变量序列和图表符号残片

    if axis_word_count >= 2 or axis_word_count >= 1 and decimal_count >= 2:  # 判断当前行是否为明显坐标轴标签或标签与小数刻度混合
        return True  # 过滤 Actual、Predicted、tested 等图表残片

    if decimal_count >= 4 and letter_count < 20:  # 判断当前行是否包含大量小数但自然语言很少
        return True  # 过滤散点图刻度、表格数值行和预测值序列

    return False  # 其余行视为可用自然语言文本


def clean_extracted_text(text: str) -> str:  # 定义 PDF 页面文本清洗函数，保留正文并移除行级噪声
    cleaned_lines = []  # 创建清洗后行列表，用于保留段落边界

    for raw_line in text.splitlines():  # 按原始换行逐行处理页面文本
        normalized_line = re.sub(r"[ \t]+", " ", raw_line).strip()  # 合并行内空白并清理首尾空白

        if not normalized_line:  # 判断当前行是否为空行
            if cleaned_lines and cleaned_lines[-1] != "":  # 判断是否需要保留一个段落分隔空行
                cleaned_lines.append("")  # 保留单个空行作为段落边界
            continue  # 跳过空行的后续噪声判断

        if is_noise_line(normalized_line):  # 判断当前非空行是否为图表、公式或数字噪声
            continue  # 丢弃噪声行并继续处理下一行

        cleaned_lines.append(normalized_line)  # 保存通过质量判断的自然语言文本行

    return "\n".join(cleaned_lines).strip()  # 拼接清洗后的页面文本并移除首尾空白


def load_document_text(file_path: Path) -> str:  # 定义统一文档读取函数，支持文本文件和文本型 PDF
    suffix = file_path.suffix.lower()  # 获取小写文件扩展名，兼容大写或混合大小写后缀

    if suffix in {".txt", ".md"}:  # 判断文档是否为现有支持的纯文本或 Markdown 文件
        return file_path.read_text(encoding="utf-8")  # 使用 UTF-8 编码读取并返回文本内容

    if suffix == ".pdf":  # 判断文档是否为需要逐页提取文字的 PDF 文件
        page_sections = []  # 创建页面文本列表，用于保存清洗后仍有正文的页面

        with fitz.open(file_path) as document:  # 打开 PDF 并确保读取结束后自动释放文件资源
            for page_number, page in enumerate(document, start=1):  # 从第一页开始逐页遍历 PDF
                extracted_text = page.get_text("text", sort=True)  # 按阅读顺序提取当前页原始文字
                references_match = re.search(r"(?im)^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?references\s*$", extracted_text)  # 查找独立成行且可带章节编号的参考文献标题
                reached_references = references_match is not None  # 记录当前页是否已经进入参考文献章节

                if references_match is not None:  # 判断当前页是否包含参考文献章节起点
                    extracted_text = extracted_text[:references_match.start()]  # 仅保留参考文献标题之前可能存在的正文内容

                cleaned_text = clean_extracted_text(extracted_text)  # 清洗当前页图表、公式、坐标轴和数字噪声

                if cleaned_text:  # 判断当前页清洗后是否仍有可用自然语言文本
                    page_sections.append(f"[Page {page_number}]\n{cleaned_text}")  # 为有效页面添加真实 PDF 页码标记

                if reached_references:  # 判断是否已经到达论文主文末尾的参考文献章节
                    break  # 停止读取后续参考文献和附录图表，避免其进入核心方法索引

        if not page_sections:  # 检查整个 PDF 是否都没有清洗出可用文字
            raise ValueError(PDF_NO_TEXT_MESSAGE)  # 抛出清晰提示，说明扫描版或纯图片 PDF 需要 OCR

        return "\n\n".join(page_sections)  # 使用空行连接有效页面并返回带页码标记的完整文本

    raise ValueError(f"Unsupported document type: {suffix}")  # 对不支持的文件类型抛出明确错误
