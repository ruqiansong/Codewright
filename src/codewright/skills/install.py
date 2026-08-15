"""Restricted remote Skill download and atomic local installation."""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import parse_qs, quote, urljoin, urlsplit

import httpx

from codewright.skills.models import SkillSource
from codewright.skills.parser import SkillParseError, parse_skill_file

MAX_REDIRECTS = 3
TOTAL_TIMEOUT_SECONDS = 60.0
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_FILES = 64
MAX_DEPTH = 4
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
_ALLOWED_FETCH_HOSTS = frozenset(
    {"skills.sh", "github.com", "api.github.com", "raw.githubusercontent.com"}
)


class SkillInstallError(RuntimeError):
    """A stable, safe remote installation failure."""

    def __init__(self, code: str, safe_message: str) -> None:
        if not code.strip() or not safe_message.strip():
            raise ValueError("code and safe_message must not be empty")
        self.code = code
        self.safe_message = safe_message
        super().__init__(safe_message)


@dataclass(frozen=True, slots=True)
class InstallResult:
    """Metadata for one successfully committed user Skill."""

    name: str
    target_dir: Path


@dataclass(frozen=True, slots=True)
class _Source:
    kind: Literal["skills_sh", "github_tree", "raw"]
    owner: str
    repo: str
    ref: str = ""
    path: str = ""
    skill: str = ""
    url: str = ""


class SkillInstaller:
    """Download one allowlisted remote Skill into the user Skill directory."""

    def __init__(
        self,
        user_skills_dir: str | Path,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._user_dir = Path(user_skills_dir).resolve()
        self._client = client

    async def install(self, url: str) -> InstallResult:
        """Download, validate, and commit one Skill without overwriting."""
        source = parse_install_url(url)
        try:
            async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
                files = await self._download(source)
                return await asyncio.to_thread(self._commit, files)
        except TimeoutError:
            raise SkillInstallError("download_timeout", "Skill download timed out.") from None
        except httpx.HTTPError:
            raise SkillInstallError("download_failed", "Skill download failed.") from None

    async def _download(self, source: _Source) -> tuple[tuple[str, bytes], ...]:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=None, follow_redirects=False)
        try:
            if source.kind == "raw":
                return (("SKILL.md", await _fetch_bytes(client, source.url)),)
            if source.kind == "skills_sh":
                url = (
                    "https://skills.sh/api/v1/skills/"
                    f"{quote(source.owner)}/{quote(source.repo)}/{quote(source.skill)}"
                )
                payload = await _fetch_json(client, url)
                return _skills_sh_files(payload)
            return await _github_files(client, source)
        finally:
            if owns_client:
                await client.aclose()

    def _commit(self, files: tuple[tuple[str, bytes], ...]) -> InstallResult:
        validated = _validate_files(files)
        parent = self._user_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = parent / f".{self._user_dir.name}.install-{uuid.uuid4().hex}"
        try:
            staging.mkdir(mode=0o700)
            for relative, content in validated:
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            skill_file = staging / "SKILL.md"
            try:
                skill = parse_skill_file(skill_file, SkillSource.USER, is_directory=True)
            except SkillParseError:
                raise SkillInstallError(
                    "invalid_skill",
                    "Downloaded content is not a valid Skill.",
                ) from None
            self._user_dir.mkdir(parents=True, exist_ok=True)
            target = self._user_dir / skill.name
            if target.exists() or target.is_symlink():
                raise SkillInstallError(
                    "skill_exists",
                    f"Skill already exists: {skill.name}",
                )
            try:
                os.rename(staging, target)
            except FileExistsError:
                raise SkillInstallError(
                    "skill_exists",
                    f"Skill already exists: {skill.name}",
                ) from None
            return InstallResult(skill.name, target)
        except SkillInstallError:
            raise
        except OSError:
            raise SkillInstallError("install_failed", "Skill could not be installed.") from None
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)


def parse_install_url(url: str) -> _Source:
    """Parse one of the three supported public HTTPS URL shapes."""
    if not isinstance(url, str) or not url.strip() or url != url.strip():
        raise SkillInstallError("invalid_url", "url must be a non-empty trimmed string.")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.fragment
        or parsed.query
        or parsed.hostname not in {"skills.sh", "github.com", "raw.githubusercontent.com"}
    ):
        raise SkillInstallError("invalid_url", "Unsupported Skill URL.")
    parts = _url_parts(parsed.path)
    host = parsed.hostname
    if host == "skills.sh" and len(parts) == 3:
        _validate_identifiers(parts)
        return _Source("skills_sh", parts[0], parts[1], skill=parts[2])
    if host == "github.com" and len(parts) >= 5 and parts[2] == "tree":
        owner, repo, _, ref, *path = parts
        _validate_identifiers((owner, repo, ref))
        _validate_url_path("/".join(path))
        return _Source("github_tree", owner, repo, ref=ref, path="/".join(path))
    if host == "raw.githubusercontent.com" and len(parts) >= 4:
        owner, repo, ref, *path = parts
        _validate_identifiers((owner, repo, ref))
        relative = "/".join(path)
        _validate_url_path(relative)
        if path[-1] != "SKILL.md":
            raise SkillInstallError("invalid_url", "Raw URL must point to SKILL.md.")
        return _Source("raw", owner, repo, ref=ref, path=relative, url=url)
    raise SkillInstallError("invalid_url", "Unsupported Skill URL.")


async def _fetch_response(client: httpx.AsyncClient, url: str) -> httpx.Response:
    current = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        _validate_fetch_url(current)
        response = await client.get(current, follow_redirects=False)
        if response.is_redirect:
            if redirect_count == MAX_REDIRECTS:
                raise SkillInstallError("too_many_redirects", "Too many download redirects.")
            location = response.headers.get("location")
            if not location:
                raise SkillInstallError("download_failed", "Skill download failed.")
            current = urljoin(current, location)
            continue
        if response.status_code < 200 or response.status_code >= 300:
            raise SkillInstallError(
                "http_error",
                f"Skill download returned HTTP {response.status_code}.",
            )
        return response
    raise SkillInstallError("too_many_redirects", "Too many download redirects.")


async def _fetch_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    response = await _fetch_response(client, url)
    content = response.content
    if len(content) > MAX_FILE_BYTES:
        raise SkillInstallError("file_too_large", "A Skill file exceeds the size limit.")
    return content


async def _fetch_json(client: httpx.AsyncClient, url: str) -> object:
    response = await _fetch_response(client, url)
    if len(response.content) > MAX_TOTAL_BYTES * 2:
        raise SkillInstallError("file_too_large", "A download response exceeds the size limit.")
    try:
        return response.json()
    except ValueError:
        raise SkillInstallError(
            "invalid_response",
            "Skill service returned invalid data.",
        ) from None


def _skills_sh_files(payload: object) -> tuple[tuple[str, bytes], ...]:
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise SkillInstallError("invalid_response", "Skill service returned invalid data.")
    files: list[tuple[str, bytes]] = []
    for item in payload["files"]:
        if not isinstance(item, dict):
            raise SkillInstallError("invalid_response", "Skill service returned invalid data.")
        path, content = item.get("path"), item.get("content")
        if (
            item.get("type", "file") != "file"
            or not isinstance(path, str)
            or not isinstance(content, str)
        ):
            raise SkillInstallError("unsafe_file", "Skill contains an unsupported file entry.")
        files.append((path, content.encode("utf-8")))
    return tuple(files)


async def _github_files(
    client: httpx.AsyncClient,
    source: _Source,
) -> tuple[tuple[str, bytes], ...]:
    files: list[tuple[str, bytes]] = []
    pending = [source.path]
    entry_count = 0
    while pending:
        path = pending.pop()
        api_url = (
            f"https://api.github.com/repos/{quote(source.owner)}/{quote(source.repo)}"
            f"/contents/{quote(path, safe='/')}?ref={quote(source.ref)}"
        )
        payload = await _fetch_json(client, api_url)
        entries = payload if isinstance(payload, list) else [payload]
        for item in entries:
            entry_count += 1
            if entry_count > MAX_FILES:
                raise SkillInstallError("too_many_files", "Skill contains too many files.")
            if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                raise SkillInstallError("invalid_response", "GitHub returned invalid data.")
            kind = item["type"]
            item_path = item.get("path")
            if not isinstance(item_path, str):
                raise SkillInstallError("invalid_response", "GitHub returned invalid data.")
            if kind == "dir":
                _relative_github_path(item_path, source.path)
                pending.append(item_path)
                continue
            if kind != "file" or item.get("submodule_git_url") is not None:
                raise SkillInstallError("unsafe_file", "Skill contains an unsupported file entry.")
            relative = _relative_github_path(item_path, source.path)
            encoded = item.get("content")
            if isinstance(encoded, str) and item.get("encoding") == "base64":
                try:
                    content = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError):
                    raise SkillInstallError(
                        "invalid_response", "GitHub returned invalid file content."
                    ) from None
            else:
                download_url = item.get("download_url")
                if not isinstance(download_url, str):
                    raise SkillInstallError("invalid_response", "GitHub returned invalid data.")
                content = await _fetch_bytes(client, download_url)
            files.append((relative, content))
            if len(files) > MAX_FILES:
                raise SkillInstallError("too_many_files", "Skill contains too many files.")
    return tuple(files)


def _validate_files(files: tuple[tuple[str, bytes], ...]) -> tuple[tuple[str, bytes], ...]:
    if not files or len(files) > MAX_FILES:
        raise SkillInstallError("too_many_files", "Skill file count is invalid.")
    seen: set[str] = set()
    total = 0
    skill_files = 0
    validated: list[tuple[str, bytes]] = []
    for relative, content in files:
        _validate_relative_path(relative)
        if relative in seen:
            raise SkillInstallError("duplicate_file", "Skill contains duplicate file paths.")
        seen.add(relative)
        if not isinstance(content, bytes):
            raise SkillInstallError("invalid_response", "Skill file content is invalid.")
        if len(content) > MAX_FILE_BYTES:
            raise SkillInstallError("file_too_large", "A Skill file exceeds the size limit.")
        total += len(content)
        if total > MAX_TOTAL_BYTES:
            raise SkillInstallError("skill_too_large", "Skill exceeds the total size limit.")
        if PurePosixPath(relative).name == "SKILL.md":
            skill_files += 1
        validated.append((relative, content))
    if skill_files != 1 or "SKILL.md" not in seen:
        raise SkillInstallError(
            "invalid_skill_layout",
            "Skill must contain exactly one root SKILL.md.",
        )
    return tuple(validated)


def _validate_fetch_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.fragment
        or parsed.hostname not in _ALLOWED_FETCH_HOSTS
    ):
        raise SkillInstallError("unsafe_redirect", "Download redirect is not allowed.")
    if parsed.query and parsed.hostname != "api.github.com":
        raise SkillInstallError("unsafe_redirect", "Download redirect is not allowed.")
    if parsed.hostname == "api.github.com":
        query = parse_qs(parsed.query, keep_blank_values=True)
        if set(query) != {"ref"} or len(query["ref"]) != 1 or not query["ref"][0]:
            raise SkillInstallError("unsafe_redirect", "Download redirect is not allowed.")


def _url_parts(path: str) -> list[str]:
    if not path.startswith("/") or path.endswith("/") or "//" in path:
        raise SkillInstallError("invalid_url", "Unsupported Skill URL.")
    parts = path[1:].split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise SkillInstallError("invalid_url", "Unsupported Skill URL.")
    return parts


def _validate_identifiers(values: tuple[str, ...] | list[str]) -> None:
    if any(_IDENTIFIER.fullmatch(value) is None for value in values):
        raise SkillInstallError("invalid_url", "Unsupported Skill URL.")


def _validate_url_path(path: str) -> None:
    try:
        _validate_relative_path(path)
    except SkillInstallError:
        raise SkillInstallError("invalid_url", "Unsupported Skill URL.") from None


def _validate_relative_path(path: str) -> None:
    if (
        not isinstance(path, str)
        or not path
        or "\\" in path
        or "%" in path
        or any(ord(character) < 32 for character in path)
        or path.startswith("/")
        or path.endswith("/")
        or "//" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise SkillInstallError("unsafe_path", "Skill contains an unsafe file path.")
    pure = PurePosixPath(path)
    parts = pure.parts
    if pure.is_absolute():
        raise SkillInstallError("unsafe_path", "Skill contains an unsafe file path.")
    if len(parts) > MAX_DEPTH:
        raise SkillInstallError("path_too_deep", "Skill file path exceeds the depth limit.")


def _relative_github_path(item_path: str, root: str) -> str:
    prefix = root.rstrip("/") + "/"
    if not item_path.startswith(prefix):
        raise SkillInstallError("unsafe_path", "GitHub returned an out-of-tree path.")
    relative = item_path[len(prefix) :]
    _validate_relative_path(relative)
    return relative


__all__ = [
    "InstallResult",
    "MAX_DEPTH",
    "MAX_FILE_BYTES",
    "MAX_FILES",
    "MAX_REDIRECTS",
    "MAX_TOTAL_BYTES",
    "SkillInstallError",
    "SkillInstaller",
    "TOTAL_TIMEOUT_SECONDS",
    "parse_install_url",
]
