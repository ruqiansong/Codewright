"""Dynamic reminder messages that are excluded from persistent conversation history."""

PLAN_MODE_REMINDER = """You are currently in PLAN MODE. Use only read_file, glob, and
grep to investigate. Do not write or edit files and do not execute shell commands.
Produce a clear step-by-step implementation plan, then stop and wait for the user to
approve it with /do."""

PLAN_MODE_REMINDER_CONCISE = """Remain in PLAN MODE. Use only read_file, glob, and grep.
Do not modify files or run shell commands; continue planning and wait for /do."""

EXECUTE_DIRECTIVE = "请按上面的计划开始执行。"


def system_reminder(body: str) -> str:
    """Wrap one non-empty dynamic instruction in a recognizable reminder tag."""
    normalized = body.strip()
    if not normalized:
        raise ValueError("reminder body must not be empty")
    return f"<system-reminder>\n{normalized}\n</system-reminder>"


def plan_reminder(*, full: bool) -> str:
    """Return the full or concise planning reminder, wrapped for request injection."""
    body = PLAN_MODE_REMINDER if full else PLAN_MODE_REMINDER_CONCISE
    return system_reminder(body)


__all__ = [
    "EXECUTE_DIRECTIVE",
    "PLAN_MODE_REMINDER",
    "PLAN_MODE_REMINDER_CONCISE",
    "plan_reminder",
    "system_reminder",
]
