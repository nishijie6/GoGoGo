"""Tkinter dialog for configuring optional KataGo files."""

from __future__ import annotations

import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from .katago import KataGoSettings, settings_file_path


KATAGO_RELEASES_URL = "https://github.com/lightvector/KataGo/releases"
KATAGO_NETWORKS_URL = "https://katagotraining.org/"
KATAGO_HUMAN_MODEL_URL = (
    "https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md"
    "#human-sl-analysis-guide"
)


class KataGoSettingsDialog:
    """A small path-selection dialog with file-level readiness checks."""

    def __init__(
        self,
        parent: tk.Misc,
        on_saved: Optional[Callable[[KataGoSettings], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        self.parent = parent
        self.on_saved = on_saved
        self.on_close = on_close
        self.window = tk.Toplevel(parent)
        self.window.title("KataGo 设置 · 弈境")
        self.window.configure(bg="#f2eee5")
        self.window.transient(parent)
        self.window.resizable(True, False)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _event: self.close())

        settings = KataGoSettings.load()
        self.executable_var = tk.StringVar(value=settings.executable)
        self.model_var = tk.StringVar(value=settings.model)
        self.human_model_var = tk.StringVar(value=settings.human_model)
        self.status_var = tk.StringVar()

        self._build()
        self._refresh_status()
        self.window.update_idletasks()
        width = min(820, max(670, self.window.winfo_reqwidth()))
        height = min(610, max(470, self.window.winfo_reqheight()))
        x = max(0, parent.winfo_rootx() + (parent.winfo_width() - width) // 2)
        y = max(0, parent.winfo_rooty() + (parent.winfo_height() - height) // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.grab_set()
        self.window.focus_set()

    @property
    def is_alive(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    def lift(self) -> None:
        if self.is_alive:
            self.window.deiconify()
            self.window.lift()
            self.window.focus_set()

    def _build(self) -> None:
        shell = ttk.Frame(self.window, style="Panel.TFrame", padding=20)
        shell.grid(row=0, column=0, sticky="nsew")
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)

        ttk.Label(shell, text="KataGo 职业棋手模拟", style="RuleTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            shell,
            text=(
                "职业 1–9 段需要官方 KataGo 引擎和主神经网络。可选的人类风格"
                "模型会分别使用 rank_1d 至 rank_9d 棋谱策略；不安装该模型时，"
                "程序仍可按递增搜索量运行 KataGo，但不具备对应段位的人类行棋风格。"
            ),
            style="RuleIntro.TLabel",
            wraplength=730,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(10, 15))

        fields = ttk.Frame(shell, style="Panel.TFrame")
        fields.grid(row=2, column=0, sticky="ew")
        fields.columnconfigure(0, weight=1)
        self._path_field(
            fields,
            row=0,
            title="1. KataGo 可执行文件（必需）",
            variable=self.executable_var,
            command=self._browse_executable,
        )
        self._path_field(
            fields,
            row=2,
            title="2. KataGo 主网络 *.bin.gz（必需）",
            variable=self.model_var,
            command=self._browse_model,
        )
        self._path_field(
            fields,
            row=4,
            title="3. KataGo 人类风格网络 *human*.bin.gz（推荐，可选）",
            variable=self.human_model_var,
            command=self._browse_human_model,
        )

        status_shell = tk.Frame(shell, bg="#e7e0d2", padx=11, pady=9)
        status_shell.grid(row=3, column=0, sticky="ew", pady=(14, 10))
        self.status_label = tk.Label(
            status_shell,
            textvariable=self.status_var,
            bg="#e7e0d2",
            fg="#5b4b35",
            font=("Microsoft YaHei UI", 9),
            justify="left",
            anchor="w",
            wraplength=720,
        )
        self.status_label.pack(fill="x")

        ttk.Label(
            shell,
            text=(
                f"设置保存位置：{settings_file_path()}\n"
                "首次启动 OpenCL 版本时，KataGo 可能会自动调优显卡并等待较长时间。"
            ),
            style="Estimate.TLabel",
            justify="left",
            wraplength=730,
        ).grid(row=4, column=0, sticky="w", pady=(0, 12))

        downloads = ttk.Frame(shell, style="Panel.TFrame")
        downloads.grid(row=5, column=0, sticky="ew", pady=(0, 13))
        ttk.Button(
            downloads,
            text="下载 KataGo 引擎",
            style="Action.TButton",
            command=lambda: webbrowser.open(KATAGO_RELEASES_URL),
        ).grid(row=0, column=0, padx=(0, 7))
        ttk.Button(
            downloads,
            text="下载主网络",
            style="Action.TButton",
            command=lambda: webbrowser.open(KATAGO_NETWORKS_URL),
        ).grid(row=0, column=1, padx=7)
        ttk.Button(
            downloads,
            text="人类风格模型说明",
            style="Action.TButton",
            command=lambda: webbrowser.open(KATAGO_HUMAN_MODEL_URL),
        ).grid(row=0, column=2, padx=7)

        footer = ttk.Frame(shell, style="Panel.TFrame")
        footer.grid(row=6, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Button(
            footer,
            text="重新检查",
            style="Action.TButton",
            command=self._refresh_status,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            footer,
            text="取消",
            style="Action.TButton",
            command=self.close,
        ).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(
            footer,
            text="保存配置",
            style="Primary.TButton",
            command=self._save,
        ).grid(row=0, column=2, padx=(8, 0))

    def _path_field(
        self,
        parent: ttk.Frame,
        row: int,
        title: str,
        variable: tk.StringVar,
        command: Callable[[], None],
    ) -> None:
        ttk.Label(parent, text=title, style="PanelTitle.TLabel").grid(
            row=row, column=0, sticky="w", pady=(0 if row == 0 else 11, 4)
        )
        line = ttk.Frame(parent, style="Panel.TFrame")
        line.grid(row=row + 1, column=0, sticky="ew")
        line.columnconfigure(0, weight=1)
        entry = ttk.Entry(line, textvariable=variable)
        entry.grid(row=0, column=0, sticky="ew")
        entry.bind("<FocusOut>", lambda _event: self._refresh_status())
        ttk.Button(
            line,
            text="浏览…",
            style="Action.TButton",
            command=command,
        ).grid(row=0, column=1, padx=(8, 0))

    def _browse_executable(self) -> None:
        value = filedialog.askopenfilename(
            parent=self.window,
            title="选择 KataGo 可执行文件",
            filetypes=(("KataGo", "katago.exe"), ("可执行文件", "*.exe"), ("所有文件", "*.*")),
        )
        if value:
            self.executable_var.set(value)
            self._refresh_status()

    def _browse_model(self) -> None:
        value = filedialog.askopenfilename(
            parent=self.window,
            title="选择 KataGo 主神经网络",
            filetypes=(("KataGo 网络", "*.bin.gz"), ("压缩模型", "*.gz"), ("所有文件", "*.*")),
        )
        if value:
            self.model_var.set(value)
            self._refresh_status()

    def _browse_human_model(self) -> None:
        value = filedialog.askopenfilename(
            parent=self.window,
            title="选择 KataGo 人类风格神经网络",
            filetypes=(("KataGo 人类网络", "*.bin.gz"), ("压缩模型", "*.gz"), ("所有文件", "*.*")),
        )
        if value:
            self.human_model_var.set(value)
            self._refresh_status()

    def _current_settings(self) -> KataGoSettings:
        return KataGoSettings(
            executable=self.executable_var.get().strip(),
            model=self.model_var.get().strip(),
            human_model=self.human_model_var.get().strip(),
        )

    def _refresh_status(self) -> None:
        settings = self._current_settings()
        errors = settings.validation_errors()
        if errors:
            self.status_var.set("尚未就绪：" + "；".join(errors) + "。")
            self.status_label.configure(fg="#9d3d31")
        elif settings.human_style_enabled:
            self.status_var.set(
                "配置完整：将使用 KataGo 主网络分析，并用人类风格模型模拟职业 1–9 段行棋。"
            )
            self.status_label.configure(fg="#315e43")
        else:
            self.status_var.set(
                "可以运行：将使用 KataGo 主网络和递增搜索量。若希望更像对应段位真人，"
                "请再配置官方人类风格模型。"
            )
            self.status_label.configure(fg="#7b5b28")

    def _save(self) -> None:
        settings = self._current_settings()
        try:
            settings.save()
        except OSError as error:
            messagebox.showerror(
                "无法保存 KataGo 设置",
                str(error),
                parent=self.window,
            )
            return
        self._refresh_status()
        errors = settings.validation_errors()
        if errors:
            messagebox.showwarning(
                "KataGo 尚未就绪",
                "配置已经保存，但还不能启动：\n\n" + "\n".join(f"• {item}" for item in errors),
                parent=self.window,
            )
            return
        if self.on_saved is not None:
            self.on_saved(settings)
        self.close()

    def close(self) -> None:
        try:
            self.window.grab_release()
            self.window.destroy()
        except tk.TclError:
            pass
        if self.on_close is not None:
            callback = self.on_close
            self.on_close = None
            callback()
