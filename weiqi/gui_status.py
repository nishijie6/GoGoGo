"""Side-panel status, win-rate bar, move list and game-over notice for :class:`GoApp`."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from typing import Optional

from .ai import AIMove, is_human_sl_difficulty
from .engine import BLACK, WHITE, MoveRecord, color_name
from .gui_constants import COLUMN_NAMES, MODE_AI
from .winrate import WinRateEstimate


class StatusPanel:
    """Mixed into :class:`~weiqi.gui.GoApp`; relies on the widgets, Tk
    variables and game state it creates, and on its other mixins."""

    def _refresh(self, notice: Optional[str] = None) -> None:
        if self.in_reasoning_mode and self.game.game_over:
            result = self.game.result_text or "当前变化已经结束"
            self.notice_var.set(
                f"推演分支已结束：{result}。可悔棋继续推演，"
                "或退出推理模式恢复正式棋局。"
            )
        elif self.game.game_over and self.game.result_text:
            self.notice_var.set(self.game.result_text)
        elif notice is not None:
            self.notice_var.set(notice)

        if self.in_reasoning_mode:
            if self.game.game_over:
                self.turn_var.set("◆ 推理模式 · 分支结束")
            else:
                self.turn_var.set(
                    f"◆ 推理模式 · {color_name(self.game.current_player)}推演"
                )
        elif self.analysis_active:
            self.turn_var.set("◇ AI 分析工作台 · 正式棋局已暂停")
        elif self.game.game_over:
            self.turn_var.set("对局结束")
        elif self.ai_busy:
            self.turn_var.set(
                f"● {color_name(self.game.current_player)} · 电脑 · "
                f"{self.active_difficulty}"
            )
        elif self.active_mode == MODE_AI:
            role = (
                "玩家"
                if self.game.current_player == self.human_color
                else f"电脑 · {self.active_difficulty}"
            )
            self.turn_var.set(f"● {color_name(self.game.current_player)} · {role}")
        else:
            self.turn_var.set(f"● {color_name(self.game.current_player)}落子")

        self.capture_var.set(
            f"提子：黑 {self.game.captures[BLACK]}  ·  白 {self.game.captures[WHITE]}"
        )
        session = self._reasoning_session
        if session is not None:
            self.move_var.set(
                f"正式 {session.start_move_number} 手  ·  "
                f"推演 +{session.variation_move_count} 手  ·  "
                f"连续虚手 {self.game.consecutive_passes}"
            )
        else:
            self.move_var.set(
                f"手数：{self.game.move_number}  ·  连续虚手：{self.game.consecutive_passes}"
            )
        if self.game.moves:
            last = self.game.moves[-1]
            if last.kind == "play" and last.row is not None and last.col is not None:
                last_text = self._coordinate(last.row, last.col)
            elif last.kind == "pass":
                last_text = "虚手"
            else:
                last_text = "认输"
            self.last_var.set(f"上一手：{color_name(last.color)} {last_text}")
        else:
            self.last_var.set("上一手：—")

        if self.in_reasoning_mode:
            self.reasoning_button.configure(
                text="退出推理  F3",
                style="ReasoningActive.TButton",
                state="normal",
            )
        else:
            self.reasoning_button.configure(
                text="开启推理  F3",
                style="Header.TButton",
                state=(
                    "disabled"
                    if self.game.game_over or self.analysis_active
                    else "normal"
                ),
            )
        if self.analysis_active:
            self.analysis_button.configure(
                text="分析已开启  F4",
                style="ReasoningActive.TButton",
                state="normal",
            )
        else:
            self.analysis_button.configure(
                text="AI 分析  F4",
                style="Header.TButton",
                state=(
                    "disabled"
                    if self.game.game_over or self.in_reasoning_mode
                    else "normal"
                ),
            )
        undo_state = (
            "normal"
            if self.game.can_undo and not self.analysis_active
            else "disabled"
        )
        self.undo_button.configure(state=undo_state)
        action_state = "normal" if self._human_can_act() else "disabled"
        self.pass_button.configure(state=action_state)
        self.resign_button.configure(state=action_state)
        self._update_winrate()
        self._refresh_move_log()
        self.draw_board()

    def _update_winrate(self) -> None:
        cache_key = self._position_cache_key()
        if cache_key != self._winrate_cache_key:
            self._last_winrate = self.winrate_estimator.estimate(self.game)
            self._winrate_cache_key = cache_key
            self._winrate_source = "启发式估算"

        estimate = self._last_winrate
        if estimate is None:
            return
        self.winrate_var.set(
            f"黑 {estimate.black_percent:.1f}%  ·  白 {estimate.white_percent:.1f}%"
        )
        if estimate.final:
            if self.game.winner is None:
                detail = "和棋"
            else:
                detail = f"{color_name(self.game.winner)}胜"
            self.winlead_var.set(f"终局 · {detail}")
        elif abs(estimate.black_lead) < 0.35:
            self.winlead_var.set(
                f"{estimate.phase} · 局势接近均衡（{self._winrate_source}）"
            )
        else:
            leader = "黑" if estimate.black_lead > 0 else "白"
            self.winlead_var.set(
                f"{estimate.phase} · {leader}约领先 {abs(estimate.black_lead):.1f} 目"
                f"（{self._winrate_source}）"
            )
        self._draw_winrate_bar()

    def _position_cache_key(self) -> tuple[object, ...]:
        return (
            self.game.size,
            self.game.komi,
            self.game.board_hash(),
            self.game.current_player,
            self.game.move_number,
            self.game.consecutive_passes,
            self.game.game_over,
            self.game.winner,
        )

    def _store_katago_winrate(self, decision: AIMove) -> None:
        """Cache KataGo's post-move evaluation for the position now on screen."""

        probability = decision.black_win_probability
        if probability is None or self.game.game_over:
            return
        heuristic = self.winrate_estimator.estimate(self.game)
        black_lead = (
            decision.black_lead
            if decision.black_lead is not None
            else heuristic.black_lead
        )
        correction = black_lead - heuristic.black_lead
        self._last_winrate = WinRateEstimate(
            black_win_probability=max(0.0, min(1.0, probability)),
            black_expected_score=heuristic.black_expected_score + correction / 2.0,
            white_expected_score=heuristic.white_expected_score - correction / 2.0,
            black_lead=black_lead,
            phase=heuristic.phase,
        )
        self._winrate_cache_key = self._position_cache_key()
        source = "HumanSL" if is_human_sl_difficulty(self.active_difficulty) else "KataGo"
        self._winrate_source = f"{source} · {decision.analysis_visits} visits"

    def _draw_winrate_bar(self) -> None:
        if not hasattr(self, "winrate_bar"):
            return
        self.winrate_bar.delete("all")
        estimate = self._last_winrate
        if estimate is None:
            return
        width = max(
            10,
            self.winrate_bar.winfo_width(),
            self.winrate_bar.winfo_reqwidth(),
        )
        height = max(10, self.winrate_bar.winfo_height())
        black_width = width * estimate.black_win_probability
        self.winrate_bar.create_rectangle(
            0,
            0,
            black_width,
            height,
            fill="#1b201d",
            outline="",
        )
        self.winrate_bar.create_rectangle(
            black_width,
            0,
            width,
            height,
            fill="#ece7dc",
            outline="",
        )
        self.winrate_bar.create_line(
            width / 2,
            0,
            width / 2,
            height,
            fill="#8e887d",
            width=1,
        )

    def _refresh_move_log(self) -> None:
        self.move_log.delete(0, tk.END)
        session = self._reasoning_session
        for index, move in enumerate(self.game.moves, start=1):
            if session is not None and index == session.start_move_number + 1:
                self.move_log.insert(tk.END, "──── 推理分支起点 ────")
            self.move_log.insert(tk.END, self._format_move(index, move))
        if session is not None and self.game.move_number == session.start_move_number:
            self.move_log.insert(tk.END, "──── 推理分支起点 ────")
        if self.game.moves:
            self.move_log.see(tk.END)

    def _format_move(self, number: int, move: MoveRecord) -> str:
        stone = "●" if move.color == BLACK else "○"
        if move.kind == "play" and move.row is not None and move.col is not None:
            text = self._coordinate(move.row, move.col)
            if move.captured:
                text += f"  提 {move.captured}"
        elif move.kind == "pass":
            text = "虚手"
        else:
            text = "认输"
        return f"{number:>3}. {stone}  {text}"

    def _coordinate(self, row: int, col: int) -> str:
        return f"{COLUMN_NAMES[col]}{self.game.size - row}"

    def _show_game_over(self) -> None:
        if self.in_reasoning_mode or self._end_dialog_shown:
            return
        self._end_dialog_shown = True
        if self.game.score_result is not None:
            score = self.game.score_result
            message = (
                "双方连续虚手，对局结束。按当前盘面采用中国数子法计分：\n\n"
                f"黑方：棋子 {score.black_stones} + 围空 {score.black_territory}"
                f" = {score.black_total:g}\n"
                f"白方：棋子 {score.white_stones} + 围空 {score.white_territory}"
                f" + 贴目 {score.komi:g} = {score.white_total:g}\n\n"
                f"{self.game.result_text}\n\n"
                "提示：程序不自动判定死子，终局前应先提净死子。"
            )
        else:
            message = self.game.result_text
        messagebox.showinfo("对局结果", message, parent=self.root)
