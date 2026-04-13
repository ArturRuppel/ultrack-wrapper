"""Launch a command in a new OS terminal window."""

from __future__ import annotations

import platform
import subprocess


def launch_in_terminal(command: str) -> None:
    """Open a new OS terminal and run *command* inside it.

    The terminal stays open after the command finishes so the user can read
    the output.  Supported platforms: Linux (gnome-terminal), macOS (Terminal),
    Windows (cmd).
    """
    system = platform.system()
    if system == "Linux":
        subprocess.Popen(
            ["gnome-terminal", "--", "bash", "-c", f"{command}; exec bash"]
        )
    elif system == "Darwin":
        # Escape single quotes inside the command string for AppleScript
        escaped = command.replace("'", "'\\''")
        apple_script = f"tell application \"Terminal\" to do script '{escaped}'"
        subprocess.Popen(["osascript", "-e", apple_script])
    elif system == "Windows":
        subprocess.Popen(f'start cmd /k "{command}"', shell=True)
    else:
        raise RuntimeError(
            f"Unsupported platform '{system}'. "
            "Open a terminal manually and run:\n"
            f"  {command}"
        )
