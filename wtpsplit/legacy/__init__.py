"""Legacy WtP API retained for ACL 2023 reproducibility.

BertChar/LACanine configs and models live under this package but do not need the
optional ``skops``/``sklearn`` stack. Only the ``WtP`` entry point does.
"""

from __future__ import annotations

__all__ = ["WtP"]


def __getattr__(name: str):
    if name == "WtP":
        from importlib.util import find_spec

        if find_spec("skops") is None:
            error = ModuleNotFoundError(
                "WtP requires the legacy dependencies. Install them with `pip install 'wtpsplit[legacy]'`."
            )
            error.name = "skops"
            raise error
        from wtpsplit.legacy.wtp import WtP

        return WtP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
