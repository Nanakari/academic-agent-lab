import re  # 导入正则表达式模块，用于统计词语、页码标记和小数残片
import unicodedata  # 导入 Unicode 分类模块，用于识别数学符号和图表符号
from typing import List  # 导入 List，用于标注返回值是字符串列表


MATH_AND_CHART_SYMBOLS = set("=±×÷∑∏√∞≈≠≤≥∫∆∂∇^_{}[]|<>→←↔⊕⊗∈∉∪∩′″")  # 定义常见数学和图表符号集合
AXIS_WORDS = {"actual", "predicted", "tested"}  # 定义常见坐标轴和图表残片词语


def is_valid_chunk(chunk: str) -> bool:  # 定义 chunk 质量判断函数，用于过滤图表、公式和数值主导的低质量片段
    normalized_chunk = chunk.strip()  # 清理 chunk 首尾空白以便进行质量统计

    if len(normalized_chunk) < 60:  # 判断 chunk 是否短到难以提供足够论文证据
        return False  # 过滤内容过短的 chunk

    content_without_page_markers = re.sub(r"\[Page\s+\d+\]", " ", normalized_chunk, flags=re.IGNORECASE)  # 移除页码标记避免干扰质量统计
    visible_characters = [character for character in content_without_page_markers if not character.isspace()]  # 收集所有非空白正文字符
    visible_count = len(visible_characters)  # 计算正文可见字符数量

    if visible_count < 50:  # 判断移除页码标记后的正文是否仍然过短
        return False  # 过滤仅包含标题、页码或少量残片的 chunk

    letter_count = sum(character.isalpha() for character in visible_characters)  # 统计 Unicode 字母和中文字符数量
    digit_count = sum(character.isdigit() for character in visible_characters)  # 统计数字字符数量
    math_symbol_count = sum(character in MATH_AND_CHART_SYMBOLS or unicodedata.category(character) == "Sm" for character in visible_characters)  # 统计数学和图表符号数量
    digit_ratio = digit_count / visible_count  # 计算数字字符占全部正文字符的比例
    math_symbol_ratio = math_symbol_count / visible_count  # 计算数学和图表符号占全部正文字符的比例
    token_count = len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|[\u4e00-\u9fff]", content_without_page_markers))  # 统计英文词和中文字符形式的近似 token 数量
    decimal_count = len(re.findall(r"(?<!\w)[+-]?(?:\d+\.\d+|\.\d+)(?!\w)", content_without_page_markers))  # 统计独立小数数量以识别图表刻度
    word_set = {word.lower() for word in re.findall(r"[A-Za-z]+", content_without_page_markers)}  # 提取小写英文词集合用于识别坐标轴标签
    axis_word_count = len(word_set & AXIS_WORDS)  # 统计当前 chunk 命中的坐标轴特征词数量
    chunk_lines = [line.strip() for line in content_without_page_markers.splitlines() if line.strip()]  # 收集非空文本行以统计代码结构密度
    reference_year_count = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", content_without_page_markers, flags=re.IGNORECASE))  # 统计包含 2024a 等后缀形式的出版年份数量
    front_matter_signal_count = len(re.findall(r"published\s+as|department\s+of|university|@[A-Za-z0-9.-]+|\babstract\s*$", content_without_page_markers, flags=re.IGNORECASE | re.MULTILINE))  # 统计论文首页作者、单位、邮箱和摘要标题等元数据特征
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", content_without_page_markers))  # 统计完整句子数量以区分首页元数据和摘要正文
    reference_signal_count = len(re.findall(r"\bet\s+al\.|\barxiv\b|\bconference\b|\bproceedings\b|\btransactions\b|\bpreprint\b|\bworkshop\b|\bjournal\b|advances\s+in\s+neural\s+information\s+processing\s+systems", content_without_page_markers, flags=re.IGNORECASE))  # 统计参考文献格式特征词数量
    code_line_count = sum(bool(re.search(r"^(?:def|class|if|elif|else|for|while|return|import|from)\b|\b\w+\s*=\s*[^=]|[{}]", line)) for line in chunk_lines)  # 统计函数、条件、赋值和花括号等代码行特征
    code_line_ratio = code_line_count / len(chunk_lines) if chunk_lines else 0.0  # 计算代码特征行占全部非空行的比例

    if letter_count < 30:  # 判断自然语言字母或中文字符是否过少
        return False  # 过滤几乎没有自然语言内容的公式和图表片段

    if token_count < 12:  # 判断 chunk 是否缺少足够的自然语言 token
        return False  # 过滤仅有标题、变量或零散标签的 chunk

    if digit_count >= 12 and digit_ratio > 0.30:  # 判断 chunk 是否由大量数字主导
        return False  # 过滤数值表格、坐标轴刻度和预测值序列

    if math_symbol_count >= 8 and math_symbol_ratio > 0.15 and letter_count < 80:  # 判断 chunk 是否为数学符号密集且自然语言不足的内容
        return False  # 过滤公式堆叠和图表符号残片

    if axis_word_count >= 2 and decimal_count >= 3:  # 判断 chunk 是否同时包含坐标轴标签和大量小数
        return False  # 过滤 Actual、Predicted、tested 与刻度值混合的图表残片

    if decimal_count >= 8 and letter_count < 80:  # 判断 chunk 是否包含大量小数但正文字符很少
        return False  # 过滤散点图、折线图和表格数值残片

    if reference_year_count >= 2 and reference_signal_count >= 2:  # 判断 chunk 是否由多条参考文献记录主导
        return False  # 过滤作者、年份、会议和期刊信息密集的参考文献 chunk

    if front_matter_signal_count >= 3 and sentence_count < 3:  # 判断 chunk 是否仅包含标题、作者单位和摘要标题而缺少正文
        return False  # 过滤低句子密度的论文首页元数据 chunk

    if code_line_count >= 4 and code_line_ratio > 0.35:  # 判断 chunk 是否由附录代码或伪代码行主导
        return False  # 过滤代码实现、字典字段和控制流残片组成的 chunk

    return True  # 通过全部质量检查的 chunk 可以进入向量索引


def split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:  # 定义文本切分函数
    text = text.strip()  # 去掉文本开头和结尾的空白字符

    if not text:  # 如果文本为空
        return []  # 返回空列表

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]  # 按空行切分段落，并去掉空段落

    chunks = []  # 创建列表，用来保存最终 chunk

    current = ""  # 创建当前正在累积的 chunk 字符串

    for paragraph in paragraphs:  # 遍历每一个段落
        if len(current) + len(paragraph) + 2 <= chunk_size:  # 如果当前 chunk 加上新段落后没有超过长度限制
            if current:  # 如果当前 chunk 不是空字符串
                current += "\n\n" + paragraph  # 在当前 chunk 后追加一个空行和新段落
            else:  # 如果当前 chunk 还是空的
                current = paragraph  # 直接把当前段落作为 chunk 内容
        else:  # 如果追加新段落会超过 chunk_size
            if current:  # 如果当前 chunk 有内容
                chunks.append(current)  # 把当前 chunk 加入结果列表

            if len(paragraph) <= chunk_size:  # 如果当前段落本身不超过 chunk_size
                current = paragraph  # 把当前段落作为新的 current
            else:  # 如果当前段落本身也超过 chunk_size
                long_chunks = split_long_text(paragraph, chunk_size, overlap)  # 对长段落进行固定长度切分
                chunks.extend(long_chunks[:-1])  # 把除了最后一段之外的内容加入结果列表
                current = long_chunks[-1] if long_chunks else ""  # 把最后一段作为新的 current

    if current:  # 如果循环结束后 current 还有内容
        chunks.append(current)  # 把最后一个 chunk 加入结果列表

    return [chunk for chunk in chunks if is_valid_chunk(chunk)]  # 在返回前过滤低质量 chunk 并保留原有切分顺序


def split_long_text(text: str, chunk_size: int, overlap: int) -> List[str]:  # 定义长文本切分函数
    chunks = []  # 创建列表，用于保存长文本切分结果

    start = 0  # 设置当前切分起点

    text_length = len(text)  # 计算文本总长度

    while start < text_length:  # 当起点还没有超过文本长度时继续切分
        end = min(start + chunk_size, text_length)  # 计算当前 chunk 的结束位置

        chunk = text[start:end].strip()  # 截取当前 chunk，并去掉首尾空白

        if chunk:  # 如果 chunk 非空
            chunks.append(chunk)  # 把 chunk 加入结果列表

        if end == text_length:  # 如果已经切到文本结尾
            break  # 跳出循环

        start = max(end - overlap, start + 1)  # 设置下一个 chunk 的起点，并保留 overlap 重叠内容

    return chunks  # 返回长文本切分结果