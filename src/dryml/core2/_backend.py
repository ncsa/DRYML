from __future__ import annotations

def install_backend_method(cls: type, name: str, fn) -> None:
    old = getattr(cls, name, None)
    if old is not None and old is not fn:
        raise RuntimeError(f"{cls.__name__}.{name} already installed.")
    setattr(cls, name, fn)
