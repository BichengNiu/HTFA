# -*- coding: utf-8 -*-
"""Utilities for managing the navigation button state."""

import streamlit as st
import logging
import time
from collections.abc import Iterable
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def optimize_button_state_management(
    main_module_options: List[str],
    current_main_module: str,
    cache_duration: float = 1.0,
) -> Dict[str, str]:
    """
    Calculate button states for the available main modules.

    The cache_duration parameter is kept for API compatibility and is validated to
    ensure callers provide a positive number.
    """
    if cache_duration <= 0:
        raise ValueError("cache_duration must be a positive number.")
    if isinstance(main_module_options, (str, bytes)):
        raise TypeError("main_module_options must be an iterable of module names.")
    if not isinstance(main_module_options, Iterable):
        raise TypeError("main_module_options must be an iterable of module names.")

    return {
        module: get_button_state_for_module(module, current_main_module)
        for module in main_module_options
    }


def get_button_state_for_module(module: str, current_module: str) -> str:
    """Return the visual state for a module selection button."""
    return "primary" if module == current_module else "secondary"


def clear_button_state_cache() -> None:
    """Remove cached button state information."""
    cache_keys = [
        "ui.button_state_cache",
        "ui.button_state_time",
        "navigation.cache",
        "ui.navigation_cache",
        "button_state_cache",
        "button_state_time",
        "module_selector_cache",
        "ui_button_cache",
        "sidebar_cache",
    ]

    for key in cache_keys:
        if key in st.session_state:
            del st.session_state[key]

    logger.debug("Button state cache cleared.")


def get_cached_button_states() -> Dict[str, Any]:
    """Return cached button state information."""
    cache_data = st.session_state.get("ui.button_state_cache")
    cache_time = st.session_state.get("ui.button_state_time")

    if cache_time is not None and not isinstance(cache_time, (int, float)):
        raise TypeError("Cached timestamp must be numeric.")

    return {
        "cache": cache_data,
        "timestamp": cache_time,
        "exists": cache_data is not None and cache_time is not None,
    }


def update_button_state_cache(
    main_module_options: List[str],
    current_main_module: str,
) -> None:
    """Recompute and persist button state information."""
    button_states = optimize_button_state_management(
        main_module_options,
        current_main_module,
    )
    timestamp = time.time()

    st.session_state["ui.button_state_cache"] = button_states
    st.session_state["ui.button_state_time"] = timestamp

    logger.debug("Button state cache updated.")


def get_button_states_for_modules(
    modules: List[str],
    current_module: str,
) -> Dict[str, str]:
    """Convenience helper that returns button states for specific modules."""
    return {
        module: get_button_state_for_module(module, current_module)
        for module in modules
    }


__all__ = [
    "optimize_button_state_management",
    "get_button_state_for_module",
    "clear_button_state_cache",
    "get_cached_button_states",
    "update_button_state_cache",
    "get_button_states_for_modules",
]
