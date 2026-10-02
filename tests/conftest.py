from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--acceptance",
        action="store_true",
        default=False,
        help="include slower architectural acceptance tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--acceptance"):
        return
    skip_acceptance = pytest.mark.skip(
        reason="acceptance test; rerun with --acceptance"
    )
    for item in items:
        if "acceptance" in item.keywords:
            item.add_marker(skip_acceptance)
