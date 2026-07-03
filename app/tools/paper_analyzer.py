"""Lightweight paper analysis without requiring a remote LLM."""

import re


class PaperAnalyzer:
    """Extract research components from paper text using section and cue matching."""

    CUES = {
        "problem": ("abstract", "problem", "challenge", "motivation", "研究问题", "挑战"),
        "method": ("method", "approach", "framework", "propose", "方法", "框架"),
        "experiment": ("experiment", "evaluation", "dataset", "benchmark", "实验", "数据集"),
        "limitation": ("limitation", "future work", "however", "局限", "不足"),
    }

    def summarize_paper_text(self, text: str, max_sentences: int = 5) -> str:
        sentences = self._sentences(text)
        selected = []
        for sentence in sentences:
            lowered = sentence.casefold()
            if any(cue in lowered for cues in self.CUES.values() for cue in cues):
                selected.append(sentence)
            if len(selected) >= max_sentences:
                break
        if not selected:
            selected = sentences[:max_sentences]
        return " ".join(selected).strip()

    def extract_problem_method_experiment_limitation(self, text: str) -> dict:
        sentences = self._sentences(text)
        result = {}
        for category, cues in self.CUES.items():
            matches = [
                sentence for sentence in sentences
                if any(cue in sentence.casefold() for cue in cues)
            ]
            result[category] = matches[:3] or ["Not explicitly stated in the local evidence."]
        return result

    @staticmethod
    def _sentences(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?。！？])\s+", normalized)
            if len(sentence.strip()) >= 20
        ]
