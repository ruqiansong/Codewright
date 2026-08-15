"""Side-effecting tool for restricted remote Skill installation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

from codewright.skills import SkillDef, SkillInstaller, SkillInstallError, SkillLoader
from codewright.tool.models import Result

type RefreshCallback = Callable[[tuple[SkillDef, ...]], None]


class InstallSkillTool:
    """Install one allowlisted remote Skill and refresh runtime discovery."""

    name = "install_skill"
    read_only = False
    description = "Install a Skill from a supported HTTPS skills.sh or GitHub URL."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Supported HTTPS Skill URL to install.",
            }
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(self, installer: SkillInstaller, loader: SkillLoader) -> None:
        if not isinstance(installer, SkillInstaller):
            raise TypeError("installer must be a SkillInstaller")
        if not isinstance(loader, SkillLoader):
            raise TypeError("loader must be a SkillLoader")
        self._installer = installer
        self._loader = loader
        self._refresh: RefreshCallback | None = None

    def set_refresh_callback(self, callback: RefreshCallback) -> None:
        """Inject the App's atomic command and prompt refresh callback."""
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._refresh = callback

    async def execute(self, arguments_json: str) -> Result:
        """Validate arguments, install, reload, and publish the new catalog."""
        url_or_error = _parse_url(arguments_json)
        if isinstance(url_or_error, Result):
            return url_or_error
        try:
            installed = await self._installer.install(url_or_error)
        except SkillInstallError as error:
            return _error(error.code, error.safe_message)
        try:
            skills = self._loader.reload()
            if self._refresh is None:
                raise RuntimeError("refresh callback missing")
            self._refresh(skills)
        except Exception:
            return _error(
                "refresh_failed",
                f"Skill installed: {installed.name}, but refresh failed. Run /skill reload.",
            )
        return Result(content=f"Skill installed: {installed.name}")


def _parse_url(arguments_json: str) -> str | Result:
    try:
        arguments = json.loads(arguments_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return _error("invalid_arguments", "Arguments must be a valid JSON object.")
    if not isinstance(arguments, dict):
        return _error("invalid_arguments", "Arguments must be a JSON object.")
    if set(arguments) != {"url"}:
        return _error("invalid_arguments", "Exactly one url argument is required.")
    url = arguments["url"]
    if not isinstance(url, str) or not url.strip() or url != url.strip():
        return _error("invalid_arguments", "url must be a non-empty trimmed string.")
    return url


def _error(code: str, message: str) -> Result:
    return Result(content=message, is_error=True, error_code=code)


__all__ = ["InstallSkillTool", "RefreshCallback"]
