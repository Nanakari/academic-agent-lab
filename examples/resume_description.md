# Resume Description

## A. 精简版

**AI Scientific Agent Lab：**面向人工智能论文研究的科研智能体系统，支持本地论文解析、证据检索、结构化引用、研究 idea 生成、实验方案设计、多维验证与可复现评估。

## B. 中等版

**项目定位：**将传统论文阅读/RAG Agent 扩展为面向 AI 研究流程的
AI Scientific Agent。

**核心架构：**采用 Planner、Memory、Action Space、Verifier 四模块设计。
Planner 生成结构化研究计划；JSONL Memory 保存论文笔记、idea、实验和验证日志；
Action Space 提供 PDF/TXT/Markdown 解析、局部证据检索、idea 生成和实验设计；
Verifier 检查证据支持、历史 idea 相似度、实验完整性和可复现性。

**技术栈：**Python、dataclasses、PyMuPDF、JSONL、pytest，以及可替换的本地
关键词检索/LLM 接口。

**主要成果：**实现 section/page-aware EvidenceChunk、claim-to-evidence
引用、数据集/基线/指标/消融/风险实验计划、正负 fixture 评估和真实论文验证入口；
所有运行产物、论文与本地配置均通过 Git 忽略规则隔离。

## C. 面试讲解版

这个项目不是普通 RAG。RAG 的核心通常是“检索片段并回答问题”，而这里把检索放进
完整科研流程：先规划任务，再归纳方法和不足，生成可检验的研究假设，设计实验，最后
让多个 verifier 检查证据、实验字段和复现信息。

Planner 负责识别科研任务并生成结构化步骤；Memory 保存论文笔记、历史 idea、实验
方案和失败记录；Action Space 是论文解析、证据检索、idea 生成、实验设计和报告写入
等可执行能力；Verifier 则作为可靠性边界，防止系统把流畅生成误当成有论文支持的
结论。

EvidenceVerifier 很重要，因为它不只判断“有没有检索结果”，还把每个 idea 和关键
claim 映射到具体论文、section/page、chunk、matched keywords 和 support level。
证据不足时，报告会明确列出 Evidence Gaps 和 Unsupported Claims。

Evaluation 使用正向论文 fixture 和空语料负向 case，检查 keyword hit、
EvidenceVerifier 预期行为、实验完整度和 citation completeness。Real Paper
Validation 则允许用户把本地论文放入 `data/papers/`，生成 Agent 报告、评估 JSON
和验证摘要。

当前局限是检索仍以轻量关键词匹配为主，无法证明 idea 真正新颖，fixture 分数也不
代表真实科研能力。后续可以扩展 embedding/hybrid retrieval、BibTeX 管理、专家人工
评分、多 Agent critique，以及沙箱化代码实验执行。
