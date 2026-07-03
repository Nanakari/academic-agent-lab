"""Rule-based task classification for an offline-friendly MVP."""

import re

from app.schemas.scientific_task import ScientificTaskType


class TaskClassifier:
    """Classify explicit research requests while keeping topic-only input useful."""

    KEYWORDS = {
        ScientificTaskType.PAPER_READING: (
            "read paper", "paper reading", "summarize paper", "论文阅读", "读论文", "总结论文",
        ),
        ScientificTaskType.LITERATURE_ANALYSIS: (
            "literature", "survey", "related work", "文献综述", "文献分析", "相关工作",
        ),
        ScientificTaskType.IDEA_GENERATION: (
            "idea", "brainstorm", "创新点", "研究想法", "选题",
        ),
        ScientificTaskType.EXPERIMENT_DESIGN: (
            "experiment", "evaluation plan", "实验设计", "实验方案", "消融",
        ),
        ScientificTaskType.RESEARCH_PROPOSAL: (
            "proposal", "research plan", "研究计划", "开题", "研究方案",
        ),
    }

    def classify(self, user_query: str) -> ScientificTaskType:
        normalized = re.sub(r"\s+", " ", user_query.lower()).strip()
        scores = {
            task_type: sum(keyword in normalized for keyword in keywords)
            for task_type, keywords in self.KEYWORDS.items()
        }
        best_type = max(scores, key=scores.get)
        # A bare research direction should trigger the complete proposal workflow.
        return best_type if scores[best_type] > 0 else ScientificTaskType.RESEARCH_PROPOSAL
