from __future__ import annotations

import pytest
from test_config import provider_data

from codewright.config import Config
from codewright.coordinator import coordinator_allowed_tools, coordinator_enabled
from codewright.team.feature import fork_teammate_enabled


@pytest.mark.parametrize("field", ["enable_coordinator_mode", "enable_fork_teammate"])
@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_feature_config_fields_are_strict_booleans(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        Config.model_validate({"providers": [provider_data()], field: value})


def test_features_require_config_and_environment_double_lock() -> None:
    config = Config.model_validate(
        {
            "providers": [provider_data()],
            "enable_coordinator_mode": True,
            "enable_fork_teammate": True,
        }
    )
    assert not coordinator_enabled(config, {})
    assert coordinator_enabled(config, {"CODEWRIGHT_COORDINATOR_MODE": "true"})
    assert fork_teammate_enabled(config, {"CODEWRIGHT_FORK_TEAMMATE": "1"})


def test_coordinator_allowlist_cannot_restore_write_tools() -> None:
    allowed = coordinator_allowed_tools(
        ("read_file", "write_file", "edit_file", "TeamCreate", "TeamTaskList")
    )
    assert allowed == frozenset({"read_file", "TeamCreate", "TeamTaskList"})
