"""Tests for curated joseki and life-and-death reasoning lessons."""

from __future__ import annotations

import unittest

from weiqi.engine import BLACK, EMPTY, WHITE
from weiqi.training import (
    JOSEKI_CATEGORY,
    JOSEKI_NOTICE,
    LIFE_AND_DEATH_CATEGORY,
    TRAINING_CATEGORIES,
    TRAINING_LESSONS,
    lesson_by_title,
    lessons_for_category,
    position_after,
)


class TrainingContentTests(unittest.TestCase):
    def test_catalog_has_both_categories_and_multiple_lessons(self) -> None:
        self.assertEqual(
            TRAINING_CATEGORIES,
            (JOSEKI_CATEGORY, LIFE_AND_DEATH_CATEGORY),
        )
        self.assertGreaterEqual(len(lessons_for_category(JOSEKI_CATEGORY)), 2)
        self.assertGreaterEqual(
            len(lessons_for_category(LIFE_AND_DEATH_CATEGORY)),
            4,
        )

    def test_lesson_keys_and_titles_are_unique(self) -> None:
        keys = [lesson.key for lesson in TRAINING_LESSONS]
        labels = [(lesson.category, lesson.title) for lesson in TRAINING_LESSONS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(labels), len(set(labels)))

    def test_every_lesson_replays_legally_at_every_step(self) -> None:
        for lesson in TRAINING_LESSONS:
            with self.subTest(lesson=lesson.key):
                self.assertIn(lesson.board_size, (9, 13, 19))
                self.assertTrue(lesson.initial_stones)
                self.assertTrue(lesson.moves)
                for move_count in range(len(lesson.moves) + 1):
                    board = position_after(lesson, move_count)
                    self.assertEqual(len(board), lesson.board_size)
                    self.assertTrue(all(len(row) == lesson.board_size for row in board))
                    self.assertTrue(
                        all(value in (EMPTY, BLACK, WHITE) for row in board for value in row)
                    )

    def test_each_move_contains_prompt_hint_and_reason(self) -> None:
        for lesson in TRAINING_LESSONS:
            self.assertTrue(lesson.intro)
            self.assertTrue(lesson.objective)
            self.assertTrue(lesson.conclusion)
            for move in lesson.moves:
                with self.subTest(lesson=lesson.key, move=move.name):
                    self.assertIn(move.color, (BLACK, WHITE))
                    self.assertTrue(move.name)
                    self.assertGreaterEqual(len(move.prompt), 8)
                    self.assertGreaterEqual(len(move.hint), 8)
                    self.assertGreaterEqual(len(move.reason), 15)

    def test_joseki_notice_rejects_blind_memorization(self) -> None:
        for required in ("参考次序", "不是全盘唯一答案", "外围", "理解"):
            self.assertIn(required, JOSEKI_NOTICE)

    def test_lookup_matches_combobox_selection(self) -> None:
        for lesson in TRAINING_LESSONS:
            self.assertIs(
                lesson_by_title(lesson.category, lesson.title),
                lesson,
            )
        with self.assertRaises(KeyError):
            lesson_by_title(JOSEKI_CATEGORY, "不存在的课程")

    def test_straight_three_kill_sequence_captures_black_group(self) -> None:
        lesson = next(
            item for item in TRAINING_LESSONS if item.key == "straight_three_kill"
        )
        board = position_after(lesson, len(lesson.moves))
        black_count = sum(value == BLACK for row in board for value in row)
        self.assertEqual(black_count, 0)

    def test_move_count_bounds_are_checked(self) -> None:
        lesson = TRAINING_LESSONS[0]
        with self.assertRaises(ValueError):
            position_after(lesson, -1)
        with self.assertRaises(ValueError):
            position_after(lesson, len(lesson.moves) + 1)


if __name__ == "__main__":
    unittest.main()

