__all__ = ["main_train", "main_test"]


def __getattr__(name):
    if name in {"main_train", "main_test"}:
        from .trainer import main_test, main_train

        exports = {"main_train": main_train, "main_test": main_test}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
