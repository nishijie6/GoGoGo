"""Tests for the in-program Chinese Go rules reference."""

from __future__ import annotations

import unittest

from weiqi.rules import RULE_SECTIONS, RULES_INTRO


class RulesContentTests(unittest.TestCase):
    def test_rules_cover_the_implemented_rule_set(self) -> None:
        content = RULES_INTRO + "\n" + "\n".join(
            section.title + "\n" + "\n".join(section.paragraphs)
            for section in RULE_SECTIONS
        )
        for required_term in (
            "9×9",
            "13×13",
            "19×19",
            "气",
            "打吃",
            "提子",
            "自杀禁手",
            "全局同形",
            "虚手",
            "认输",
            "死子",
            "中国数子法",
            "6.5 目",
            "公气",
            "实时胜率",
            "KataGo",
            "不是棋力认证",
            "F1",
            "推理训练",
            "F2",
            "推理模式",
            "F3",
            "HumanSL",
        ):
            self.assertIn(required_term, content)

    def test_sections_are_ordered_and_nonempty(self) -> None:
        self.assertEqual(len(RULE_SECTIONS), 10)
        self.assertEqual(
            [section.title.split("、", maxsplit=1)[0] for section in RULE_SECTIONS],
            ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"],
        )
        self.assertTrue(all(section.paragraphs for section in RULE_SECTIONS))


if __name__ == "__main__":
    unittest.main()
