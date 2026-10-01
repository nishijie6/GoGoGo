"""Tkinter user interface for the local Go game.

:class:`GoApp` keeps game control (new game, moves, AI turns, reasoning mode,
shutdown) and combines four mixins: layout (``gui_layout``), board drawing and
input (``gui_board``), secondary windows (``gui_windows``) and the status panel
(``gui_status``).
"""

from __future__ import annotations

import threading
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from tkinter import messagebox, ttk
from typing import Optional

from .ai import (
    AI_DIFFICULTIES,
    AIMove,
    GoAI,
    is_human_sl_difficulty,
    is_katago_difficulty,
    is_training_difficulty,
)
from .analysis_gui import AnalysisWorkbenchWindow
from .engine import BLACK, WHITE, GoGame, Point, color_name
from .katago import (
    KataGoAI,
    KataGoConfigurationError,
    KataGoEngine,
    KataGoError,
    KataGoSettings,
)
from .katago_gui import KataGoSettingsDialog
from .reasoning import ReasoningSession
from .rl_activity import GameActivity
from .training_gui import ReasoningTrainer
from .winrate import WinRateEstimate, WinRateEstimator
from .gui_board import BoardView
from .gui_constants import MODE_AI, MODE_LOCAL
from .gui_layout import GoAppLayout
from .gui_status import StatusPanel
from .gui_windows import CompanionWindows
from .rl_opponents import load_training_opponents, settings_for_training_opponent




class GoApp(GoAppLayout, BoardView, CompanionWindows, StatusPanel):
    """A responsive desktop Go board supporting AI and local play."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("弈境 · 围棋")
        self.root.geometry("1080x760")
        self.root.minsize(900, 690)
        self.root.configure(bg="#172019")

        self._configure_styles()

        self.game = GoGame(size=9)
        self._reasoning_session: Optional[ReasoningSession] = None
        self.ai = GoAI()
        self.katago_engine: Optional[KataGoEngine] = None
        self.winrate_estimator = WinRateEstimator()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="go-ai")
        self.ai_future: Optional[Future[AIMove]] = None
        self._ai_cancel_event: Optional[threading.Event] = None
        self.ai_busy = False
        self.generation = 0
        self.active_mode = MODE_AI
        self.active_difficulty = "中等"
        self.human_color = BLACK
        self.ai_color = WHITE
        self.hover_point: Optional[Point] = None
        self._board_geometry: Optional[tuple[float, float, float]] = None
        self._end_dialog_shown = False
        self.rules_window: Optional[tk.Toplevel] = None
        self.training_window: Optional[ReasoningTrainer] = None
        self.analysis_window: Optional[AnalysisWorkbenchWindow] = None
        self.katago_settings_window: Optional[KataGoSettingsDialog] = None
        self._pending_katago_new_game = False
        self._pending_analysis_open = False
        self._closing = False
        self._winrate_cache_key: Optional[tuple[object, ...]] = None
        self._last_winrate: Optional[WinRateEstimate] = None
        self._winrate_source = "启发式估算"
        self._training_opponents = {}
        self.active_training_opponent = None

        self.mode_var = tk.StringVar(value=MODE_AI)
        self.size_var = tk.StringVar(value="9×9")
        self.human_color_var = tk.StringVar(value="黑方（先手）")
        self.difficulty_var = tk.StringVar(value="中等")
        self.turn_var = tk.StringVar()
        self.notice_var = tk.StringVar()
        self.capture_var = tk.StringVar()
        self.move_var = tk.StringVar()
        self.last_var = tk.StringVar()
        self.winrate_var = tk.StringVar()
        self.winlead_var = tk.StringVar()

        self._build_layout()
        self._refresh_training_opponents()
        self._bind_shortcuts()
        self.new_game()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._training_activity = GameActivity()
        self._training_heartbeat_id = None
        self._update_training_activity()

    def _update_training_activity(self) -> None:
        if self._closing:
            return
        self._training_activity.update(
            not self.game.game_over or self._reasoning_session is not None
        )
        self._training_heartbeat_id = self.root.after(1000, self._update_training_activity)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-n>", lambda _event: self.new_game())
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Key-p>", lambda _event: self.pass_turn())
        self.root.bind("<Key-P>", lambda _event: self.pass_turn())
        self.root.bind("<F1>", lambda _event: self.show_rules())
        self.root.bind("<F2>", lambda _event: self.show_training())
        self.root.bind("<F3>", lambda _event: self.toggle_reasoning_mode())
        self.root.bind("<F4>", lambda _event: self.show_analysis())

    def _bind_panel_mousewheel(self, widget: tk.Misc) -> None:
        if not isinstance(widget, (tk.Listbox, ttk.Scrollbar)):
            widget.bind("<MouseWheel>", self._on_panel_mousewheel, add="+")
        for child in widget.winfo_children():
            self._bind_panel_mousewheel(child)

    def _on_panel_mousewheel(self, event: tk.Event) -> str:
        if event.delta:
            direction = -1 if event.delta > 0 else 1
            self.panel_canvas.yview_scroll(direction * 2, "units")
        return "break"

    def _on_mode_selected(self, _event: object = None) -> None:
        state = "readonly" if self.mode_var.get() == MODE_AI else "disabled"
        self.human_combo.configure(state=state)
        self.difficulty_combo.configure(state=state)

    @property
    def in_reasoning_mode(self) -> bool:
        """Whether the board is currently showing an isolated variation."""

        return self._reasoning_session is not None

    def toggle_reasoning_mode(self) -> bool:
        """Enter reasoning mode, or discard the variation and restore the game."""

        if self.in_reasoning_mode:
            return self.exit_reasoning_mode()
        return self.enter_reasoning_mode()

    def enter_reasoning_mode(self) -> bool:
        """Save the formal game and switch the board to a temporary variation."""

        if self.analysis_active:
            assert self.analysis_window is not None
            self.analysis_window.lift()
            self.notice_var.set(
                "AI 分析工作台已经包含多分支推演；请先关闭它再进入 F3 推理模式。"
            )
            self.root.bell()
            return False
        if self.game.game_over:
            self.notice_var.set("本局已经结束，不能从终局开启推理模式。")
            self.root.bell()
            return False

        self._invalidate_ai()
        session = ReasoningSession.start(self.game)
        self._reasoning_session = session
        self.game = session.variation
        self.hover_point = None
        self._end_dialog_shown = False
        self._winrate_cache_key = None
        self._last_winrate = None
        self._winrate_source = "启发式估算"
        self._refresh(
            "推理模式已开启：正式棋局已保存。现在可为黑白双方连续推演，"
            "悔棋只撤回推演着手。"
        )
        return True

    def exit_reasoning_mode(self) -> bool:
        """Discard the temporary variation and resume the saved formal game."""

        session = self._reasoning_session
        if session is None:
            return False

        self._invalidate_ai()
        variation_moves = session.variation_move_count
        self.game = session.restore_formal_game()
        self._reasoning_session = None
        self.hover_point = None
        self._end_dialog_shown = False
        self._winrate_cache_key = None
        self._last_winrate = None
        self._winrate_source = "启发式估算"
        if variation_moves:
            notice = (
                f"已退出推理模式，丢弃当前推演分支的 {variation_moves} 手；"
                "正式棋局已恢复，可继续对局。"
            )
        else:
            notice = "已退出推理模式，正式棋局已恢复，可继续对局。"
        self._refresh(notice)
        if self._is_ai_turn():
            self.root.after(220, self._start_ai_turn)
        return True

    def _refresh_training_opponents(self, _event=None) -> None:
        size = int(self.size_var.get().split("×", maxsplit=1)[0])
        try:
            self._training_opponents = load_training_opponents(size)
        except (OSError, ValueError, KeyError) as error:
            self._training_opponents = {}
            self.notice_var.set(f"训练对手列表暂不可用：{error}")
        self.difficulty_combo.configure(values=(*AI_DIFFICULTIES, *self._training_opponents))
        selected = self.difficulty_var.get()
        if is_training_difficulty(selected) and selected not in self._training_opponents:
            self.difficulty_var.set("中等")

    def new_game(self) -> None:
        """Apply the selected options and replace the current game."""

        size = int(self.size_var.get().split("×", maxsplit=1)[0])
        selected_mode = self.mode_var.get()
        selected_difficulty = self.difficulty_var.get()

        candidate_engine: Optional[KataGoEngine] = None
        training_opponent = None
        if selected_mode == MODE_AI and is_katago_difficulty(selected_difficulty):
            settings = KataGoSettings.load()
            try:
                if is_training_difficulty(selected_difficulty):
                    training_opponent = self._training_opponents.get(selected_difficulty)
                    if training_opponent is None:
                        self.notice_var.set("训练对手已变化，请重新选择。")
                        return
                    settings = settings_for_training_opponent(settings, training_opponent, size)
                settings.require_valid()
            except KataGoConfigurationError as error:
                self.notice_var.set(f"所选电脑难度需要先配置 KataGo：{error}。")
                self.show_katago_settings(pending_new_game=True)
                return
            if (
                is_human_sl_difficulty(selected_difficulty)
                and not settings.human_style_enabled
            ):
                self.notice_var.set(
                    "HumanSL 人类段位需要先配置人类风格模型。"
                )
                self.show_katago_settings(pending_new_game=True)
                return

            if (
                self.katago_engine is not None
                and not self.katago_engine.closed
                and self.katago_engine.settings.fingerprint == settings.fingerprint
            ):
                candidate_engine = self.katago_engine
            else:
                try:
                    candidate_engine = KataGoEngine(settings)
                except KataGoError as error:
                    self.notice_var.set(f"KataGo 配置不可用：{error}。")
                    self.show_katago_settings(pending_new_game=True)
                    return
            candidate_ai = KataGoAI(candidate_engine, selected_difficulty)
        else:
            builtin_difficulty = (
                "中等" if is_katago_difficulty(selected_difficulty) else selected_difficulty
            )
            candidate_ai = GoAI(difficulty=builtin_difficulty)

        old_engine = self.katago_engine
        self._invalidate_ai()
        analysis_window = getattr(self, "analysis_window", None)
        if analysis_window is not None:
            analysis_window.close()
        self._reasoning_session = None
        if old_engine is not None and old_engine is not candidate_engine:
            old_engine.close()
        self.katago_engine = candidate_engine
        self.ai = candidate_ai
        self.active_mode = selected_mode
        self.active_difficulty = selected_difficulty
        self.active_training_opponent = training_opponent
        self.human_color = (
            BLACK if self.human_color_var.get().startswith("黑") else WHITE
        )
        self.ai_color = WHITE if self.human_color == BLACK else BLACK
        self.game = GoGame(size=size, komi=6.5)
        self.hover_point = None
        self._end_dialog_shown = False
        self._winrate_cache_key = None
        self._last_winrate = None
        self._winrate_source = "启发式估算"
        if self.active_mode == MODE_AI:
            engine_note = (
                "；首次启动引擎可能需要调优显卡"
                if is_katago_difficulty(self.active_difficulty)
                else ""
            )
            self.notice_var.set(
                f"新对局已开始，电脑难度：{self.active_difficulty}{engine_note}。"
            )
            if is_human_sl_difficulty(self.active_difficulty) and size != 19:
                self.notice_var.set(
                    self.notice_var.get()
                    + " HumanSL 级段位主要基于 19×19 人类棋谱，"
                    "当前尺寸仅作风格模拟。"
                )
        else:
            self.notice_var.set("新对局已开始，请在棋盘交叉点落子。")
        self._on_mode_selected()
        self._refresh()
        if self._is_ai_turn():
            self.root.after(300, self._start_ai_turn)

    def pass_turn(self) -> None:
        if not self._human_can_act():
            if self.ai_busy:
                self.notice_var.set("电脑正在思考，现在不能虚手。")
            return
        color = self.game.current_player
        if not self.game.pass_turn():
            return
        self.hover_point = None
        prefix = "推演：" if self.in_reasoning_mode else ""
        self._refresh(f"{prefix}{color_name(color)}选择虚手。")
        if self.game.game_over:
            if not self.in_reasoning_mode:
                self._show_game_over()
        elif self._is_ai_turn():
            self._start_ai_turn()

    def resign(self) -> None:
        if not self._human_can_act():
            return
        color = self.game.current_player
        reasoning = self.in_reasoning_mode
        title = "确认推演认输" if reasoning else "确认认输"
        prompt = (
            f"确定让{color_name(color)}在当前推演分支认输吗？"
            "这不会影响正式棋局。"
            if reasoning
            else f"确定由{color_name(color)}认输并结束本局吗？"
        )
        if not messagebox.askyesno(
            title,
            prompt,
            parent=self.root,
        ):
            return
        if self.game.resign():
            self.hover_point = None
            if reasoning:
                self._refresh(
                    f"推演分支：{self.game.result_text}。可悔棋继续推演，"
                    "或退出推理模式恢复正式棋局。"
                )
            else:
                self._refresh(self.game.result_text)
                self._show_game_over()

    def undo(self) -> None:
        if self.analysis_active:
            self.notice_var.set(
                "正式棋局已暂停；请使用 AI 分析工作台中的“撤回”浏览推演树。"
            )
            assert self.analysis_window is not None
            self.analysis_window.lift()
            return
        if not self.game.can_undo:
            if self.in_reasoning_mode:
                self.notice_var.set(
                    "推演已经回到保存点，不能撤回正式棋局中的着手。"
                )
            else:
                self.notice_var.set("当前没有可以撤回的着手。")
            return

        was_ai_busy = self.ai_busy
        self._invalidate_ai()
        undone = self.game.undo(1)
        if self.in_reasoning_mode:
            self.hover_point = None
            self._end_dialog_shown = False
            self._refresh(f"推演已撤回 {undone} 手，正式棋局保持不变。")
            return
        if self.active_mode == MODE_AI and not was_ai_busy:
            # Normally take back the AI response and the preceding human action,
            # leaving the human at the decision they wanted to reconsider.
            while self.game.can_undo and self.game.current_player != self.human_color:
                undone += self.game.undo(1)

        self.hover_point = None
        self._end_dialog_shown = False
        self._refresh(f"已撤回 {undone} 手。")
        if self._is_ai_turn():
            self.root.after(220, self._start_ai_turn)

    def _start_ai_turn(self) -> None:
        if not self._is_ai_turn() or self.ai_busy:
            return
        self.ai_busy = True
        request_generation = self.generation
        snapshot = self.game.clone()
        cancel_event = threading.Event()
        self._ai_cancel_event = cancel_event
        if isinstance(self.ai, KataGoAI):
            self.ai_future = self.executor.submit(
                self.ai.choose_move,
                snapshot,
                cancel_event,
            )
        else:
            self.ai_future = self.executor.submit(self.ai.choose_move, snapshot)
        self._refresh(
            f"{color_name(self.ai_color)}电脑正在思考…"
            f"（{self.active_difficulty}）"
        )
        self.root.after(60, lambda: self._poll_ai(request_generation))

    def _poll_ai(self, request_generation: int) -> None:
        if request_generation != self.generation:
            return
        future = self.ai_future
        if future is None:
            return
        if not future.done():
            self.root.after(60, lambda: self._poll_ai(request_generation))
            return

        self.ai_busy = False
        self.ai_future = None
        self._ai_cancel_event = None
        try:
            decision = future.result()
        except Exception as error:  # Keep the GUI usable if an AI bug occurs.
            if isinstance(error, KataGoError) or isinstance(self.ai, KataGoAI):
                self._refresh(
                    f"KataGo 计算失败，棋盘已保留：{error}。"
                    "请检查设置后重试。"
                )
                messagebox.showerror(
                    "KataGo 计算失败",
                    f"{error}\n\n棋盘没有改变。请检查引擎、模型或显卡后端设置，"
                    "保存后程序会尝试继续当前对局。",
                    parent=self.root,
                )
                self.show_katago_settings()
                return
            # A safe automatic pass hands control back instead of leaving the
            # application permanently stuck on the computer's turn.
            if self._is_ai_turn():
                self.game.pass_turn()
            self._refresh(f"电脑计算失败并已自动虚手：{error}")
            if self.game.game_over:
                self._show_game_over()
            return

        if request_generation != self.generation or not self._is_ai_turn():
            self._refresh()
            return

        color = self.game.current_player
        if decision.point is None:
            self.game.pass_turn()
            notice = f"{color_name(color)}电脑选择虚手（{decision.explanation}）。"
        else:
            row, col = decision.point
            analysis = self.game.play(row, col)
            if not analysis.legal:
                # The snapshot and live game should match.  Passing is a safe
                # fallback if they ever do not.
                self.game.pass_turn()
                notice = f"{color_name(color)}电脑选择虚手。"
            else:
                coordinate = self._coordinate(row, col)
                capture_text = (
                    f"，提掉 {analysis.captured} 子" if analysis.captured else ""
                )
                notice = (
                    f"{color_name(color)}电脑于 {coordinate} 落子{capture_text}"
                    f"（{decision.explanation}）。"
                )
        self._store_katago_winrate(decision)
        self._refresh(notice)
        if self.game.game_over:
            self._show_game_over()

    def _invalidate_ai(self) -> None:
        self.generation += 1
        cancel_event = getattr(self, "_ai_cancel_event", None)
        if cancel_event is not None:
            cancel_event.set()
        self._ai_cancel_event = None
        future_running = self.ai_future is not None and not self.ai_future.done()
        if self.ai_future is not None:
            self.ai_future.cancel()
        if future_running and self.katago_engine is not None:
            self.katago_engine.stop()
        self.ai_future = None
        self.ai_busy = False

    def _is_ai_turn(self) -> bool:
        return (
            not self.in_reasoning_mode
            and not self.analysis_active
            and self.active_mode == MODE_AI
            and not self.game.game_over
            and self.game.current_player == self.ai_color
        )

    def _human_can_act(self) -> bool:
        if self.game.game_over or self.ai_busy or self.analysis_active:
            return False
        if self.in_reasoning_mode:
            return True
        return self.active_mode == MODE_LOCAL or self.game.current_player == self.human_color

    def close(self) -> None:
        self._closing = True
        heartbeat = getattr(self, "_training_heartbeat_id", None)
        if heartbeat is not None:
            self.root.after_cancel(heartbeat)
        activity = getattr(self, "_training_activity", None)
        if activity is not None:
            activity.close()
        self._invalidate_ai()
        analysis_window = getattr(self, "analysis_window", None)
        if analysis_window is not None:
            analysis_window.close()
            self.analysis_window = None
        if self.katago_engine is not None:
            self.katago_engine.close()
            self.katago_engine = None
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.training_window is not None:
            self.training_window.close()
            self.training_window = None
        if self.katago_settings_window is not None:
            self.katago_settings_window.close()
            self.katago_settings_window = None
        self._close_rules()
        self.root.destroy()


def run() -> None:
    root = tk.Tk()
    GoApp(root)
    root.mainloop()
