"""Layered permission decisions for built-in Codewright tool calls."""

import logging
from dataclasses import dataclass
from pathlib import Path

from codewright.llm import ToolCall
from codewright.permission.blacklist import hits_blacklist
from codewright.permission.models import (
    Category,
    Decision,
    Mode,
    PermissionSetupError,
    parse_mode,
)
from codewright.permission.rule import RuleSet, is_mcp_tool_name
from codewright.permission.sandbox import (
    eval_symlinks_or_ancestor,
    resolve_root,
    sandbox_ok,
)
from codewright.permission.settings import (
    Settings,
    SettingsError,
    categorize,
    extract_target,
    friendly_name,
    load_settings,
    search_pattern_safe,
    to_rule_set,
)

logger = logging.getLogger(__name__)

_BUILTIN_TOOLS = frozenset(
    {
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "load_skill",
        "install_skill",
        "Agent",
        "TaskList",
        "TaskGet",
        "TaskStop",
        "SendMessage",
    }
)


@dataclass(slots=True)
class Engine:
    """Resolved project root, layered rules, and startup permission mode."""

    root: Path
    user: RuleSet
    project: RuleSet
    local: RuleSet
    local_path: Path
    default_mode: Mode

    def check(
        self,
        mode: Mode,
        call: ToolCall,
        read_only: bool,
    ) -> tuple[Decision, str]:
        """Apply blacklist, sandbox, explicit rules, then mode fallback."""
        if not isinstance(mode, Mode):
            return Decision.DENY, "无效的权限模式，安全拒绝"
        if call.name not in _BUILTIN_TOOLS and not is_mcp_tool_name(call.name):
            return Decision.DENY, "未知工具，安全拒绝"

        target, is_file, ok = extract_target(call)
        if not ok:
            return Decision.DENY, "工具参数无效，安全拒绝"
        try:
            category = categorize(call.name, read_only)
        except (TypeError, ValueError):
            return Decision.DENY, "无法识别工具安全类别，安全拒绝"

        if category is Category.EXEC and target and hits_blacklist(target):
            return Decision.DENY, f"命中危险命令黑名单：{_bounded(target)}"

        rule_target = target
        if is_file:
            if not search_pattern_safe(call):
                return Decision.DENY, "搜索模式包含不安全路径，安全拒绝"
            if not sandbox_ok(self.root, target):
                return Decision.DENY, f"路径在项目目录之外：{_bounded(target)}"
            try:
                rule_target = _relative_target(self.root, target)
            except (OSError, RuntimeError, ValueError):
                return Decision.DENY, "无法解析文件路径参数，安全拒绝"

        public_name = friendly_name(call.name)
        for layer in (self.local, self.project, self.user):
            decision, matched = layer.match(public_name, rule_target)
            if matched:
                if decision is Decision.DENY:
                    return decision, f"匹配 deny 规则：{public_name}({rule_target})"
                return decision, ""

        decision = mode_fallback(mode, category)
        if decision is Decision.ASK:
            return decision, f"{mode} 模式下 {_category_name(category)} 类操作需确认"
        return decision, ""

    def persist_local_allow(self, call: ToolCall) -> None:
        """Persist one exact allow rule in the project-local settings layer."""
        from codewright.permission.persist import persist_local_allow

        persist_local_allow(self, call)


def new_engine(root: Path) -> Engine:
    """Create a safe engine while degrading invalid individual settings layers."""
    try:
        resolved_root = resolve_root(root)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError) as error:
        raise PermissionSetupError("无法建立安全的项目根目录。") from error

    user_path = Path.home() / ".codewright" / "settings.yaml"
    project_path = resolved_root / ".codewright" / "settings.yaml"
    local_path = resolved_root / ".codewright" / "settings.local.yaml"
    loaded = [_load_layer(path) for path in (user_path, project_path, local_path)]
    user_settings, project_settings, local_settings = loaded

    return Engine(
        root=resolved_root,
        user=to_rule_set(user_settings),
        project=to_rule_set(project_settings),
        local=to_rule_set(local_settings),
        local_path=local_path,
        default_mode=_select_default_mode(
            local_settings,
            project_settings,
            user_settings,
        ),
    )


def mode_fallback(mode: Mode, category: Category) -> Decision:
    """Return the fixed fallback matrix after no explicit rule matched."""
    if category is Category.READ or mode is Mode.BYPASS:
        return Decision.ALLOW
    if mode is Mode.ACCEPT_EDITS and category is Category.WRITE:
        return Decision.ALLOW
    return Decision.ASK


def _load_layer(path: Path) -> Settings:
    try:
        return load_settings(path)
    except SettingsError:
        logger.warning("Ignoring invalid permission settings file: %s", path)
        return Settings()


def _select_default_mode(*settings: Settings) -> Mode:
    for item in settings:
        if not item.default_mode:
            continue
        mode, valid = parse_mode(item.default_mode)
        if valid:
            return mode
        logger.warning("Ignoring invalid default_mode in permission settings")
    return Mode.DEFAULT


def _relative_target(root: Path, target: str) -> str:
    requested = Path(target).expanduser() if target else root
    if not requested.is_absolute():
        requested = root / requested
    resolved = eval_symlinks_or_ancestor(requested)
    relative = resolved.relative_to(root)
    return relative.as_posix() or "."


def _bounded(value: str, limit: int = 160) -> str:
    compact = value.replace("\n", " ").replace("\r", " ")
    return compact if len(compact) <= limit else compact[:limit] + "…"


def _category_name(category: Category) -> str:
    return {
        Category.READ: "读取",
        Category.WRITE: "写入",
        Category.EXEC: "执行",
    }[category]


__all__ = ["Engine", "mode_fallback", "new_engine"]
