# -*- coding: utf-8 -*-
"""Streamlit page configuration helpers without silent fallbacks."""

from typing import Any, Dict, Optional

import streamlit as st


def safe_set_page_config(
    page_title: str = "经济运行分析平台",
    page_icon: str = "[CHART]",
    layout: str = "wide",
    initial_sidebar_state: str = "expanded",
    menu_items: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Configure the Streamlit page exactly once.

    Raises:
        StreamlitAPIException: when Streamlit rejects the configuration.
        ValueError: when any parameter is invalid.
    """
    if get_page_config_status():
        return True

    validated_config = _validate_config_parameters(
        page_title,
        page_icon,
        layout,
        initial_sidebar_state,
        menu_items,
    )

    st.set_page_config(**validated_config)
    _set_page_config_status(True)
    return True


def _set_page_config_status(status: bool) -> None:
    """Record whether the page configuration has already been applied."""
    st.session_state["dashboard_page_config_set"] = status


def get_page_config_status() -> bool:
    """Return True when the page configuration has already been applied."""
    return st.session_state.get("dashboard_page_config_set", False)


def reset_page_config_flag() -> None:
    """
    Reset the configuration flag in session_state.
    """
    st.session_state.pop("dashboard_page_config_set", None)


def _validate_config_parameters(
    page_title: str,
    page_icon: str,
    layout: str,
    initial_sidebar_state: str,
    menu_items: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate and return the configuration dictionary."""
    valid_layouts = {"wide", "centered"}
    if layout not in valid_layouts:
        raise ValueError(f"Invalid layout '{layout}'. Expected one of {sorted(valid_layouts)}.")

    valid_sidebar_states = {"expanded", "collapsed", "auto"}
    if initial_sidebar_state not in valid_sidebar_states:
        raise ValueError(
            f"Invalid sidebar state '{initial_sidebar_state}'. "
            f"Expected one of {sorted(valid_sidebar_states)}."
        )

    config: Dict[str, Any] = {
        "page_title": page_title,
        "page_icon": page_icon,
        "layout": layout,
        "initial_sidebar_state": initial_sidebar_state,
    }

    if menu_items is not None:
        config["menu_items"] = menu_items

    return config


def get_default_config() -> Dict[str, Any]:
    """Return the default page configuration."""
    return {
        "page_title": "经济运行分析平台",
        "page_icon": "[CHART]",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }


def set_custom_config(config: Dict[str, Any]) -> bool:
    """Apply a custom configuration on top of the defaults."""
    default_config = get_default_config()
    default_config.update(config)
    return safe_set_page_config(**default_config)


__all__ = [
    "safe_set_page_config",
    "get_page_config_status",
    "reset_page_config_flag",
    "get_default_config",
    "set_custom_config",
]
