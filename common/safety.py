# encoding:utf-8

"""
简单的输出敏感词过滤器，避免返回当代中国政治等敏感话题。
"""

import re
from typing import Tuple

from common.log import logger
from config import conf


class SensitiveFilter:
    def __init__(self):
        self._load()

    def _load(self):
        words = conf().get("sensitive_block_words", []) or []
        # 去重并清理空白
        uniq = []
        for w in words:
            if not w:
                continue
            w = str(w).strip()
            if w and w not in uniq:
                uniq.append(w)
        self.words = uniq

    def reload(self):
        self._load()

    def filter(self, text: str) -> Tuple[bool, str]:
        """
        返回 (是否通过, 可能被替换后的文本)
        """
        if not text:
            return True, text
        lowered = text.lower()
        for w in self.words:
            if not w:
                continue
            try:
                # 不区分大小写匹配
                if w.lower() in lowered:
                    logger.warning(f"[SensitiveFilter] hit sensitive word: {w}")
                    return False, "抱歉，无法提供该请求的回复。"
            except Exception as e:
                logger.debug(f"[SensitiveFilter] check word error: {e}")
                continue
        return True, text


sensitive_filter = SensitiveFilter()
