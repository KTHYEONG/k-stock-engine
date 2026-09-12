"""Point-in-time feature domain contracts and builders.

The public names are re-exported lazily so importing a lightweight consumer
does not eagerly initialize the NumPy dependency tree, mirroring
``src.strategy``.  Eager re-export makes the package import NumPy's C
extension during coverage instrumentation, which fails under pytest's
``importlib`` import mode.
"""

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "build_qvef_features": "src.features.qvef",
    "materialize_qvef_features": "src.features.materialize",
    "normalize_component_scores": "src.features.preprocessing",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "build_qvef_features",
    "materialize_qvef_features",
    "normalize_component_scores",
]
