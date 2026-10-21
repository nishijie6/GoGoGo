#!/bin/bash

# Finder starts .command files in Terminal; resolve paths relative to this file.
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)" || exit 1
cd "$PROJECT_DIR" || exit 1

show_error() {
    printf '\n启动围棋失败：%s\n\n' "$1" >&2
    if [ -t 0 ]; then
        read -r -p "按回车键关闭窗口……" launcher_reply
    fi
}

if [ ! -f "$PROJECT_DIR/main.py" ]; then
    show_error "找不到 main.py，请将启动程序保留在围棋项目目录中。"
    exit 1
fi

PYTHON_EXE="$PROJECT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_EXE" ]; then
    PYTHON_EXE="$(command -v python3 || true)"
fi

if [ -z "$PYTHON_EXE" ]; then
    show_error "未找到 Python。请先安装支持 Tk 的 Python 3.9 或更新版本。"
    exit 1
fi

if ! "$PYTHON_EXE" -c 'import sys, tkinter; sys.version_info >= (3, 9) or sys.exit(1); tkinter.Tcl()'; then
    show_error "当前 Python 不支持启动界面。请检查 Python 版本和 Tcl/Tk 安装：$PYTHON_EXE"
    exit 1
fi

export PYTHONUTF8=1
"$PYTHON_EXE" "$PROJECT_DIR/main.py"
launcher_status=$?
if [ "$launcher_status" -ne 0 ]; then
    show_error "程序未能正常运行，请查看上方的错误信息。"
fi
exit "$launcher_status"
