__all__ = [
    "ControlTrainingResult",
    "load_run_config",
    "run_control_training",
    "run_synthetic_smoke",
    "select_torch_device",
]


def __getattr__(name: str):
    if name in __all__:
        from . import control

        return getattr(control, name)
    raise AttributeError(name)
