from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn import tasks

from .cbfuse import CBLinear, CBFuse, Silence
from .dynamic_conv import C3k2_DynamicConv, DynamicConv
from .hsfpn import Add, CA, FeatureSelectionModule, multiply
from .rfa_conv import C3k2_RFAConv, RFAConv


CUSTOM_MODEL_DIR = Path(__file__).resolve().parent / "cfg" / "models" / "11"


_CUSTOM_SYMBOLS = {
    "Add": Add,
    "multiply": multiply,
    "CA": CA,
    "FeatureSelectionModule": FeatureSelectionModule,
    "Silence": Silence,
    "CBLinear": CBLinear,
    "CBFuse": CBFuse,
    "DynamicConv": DynamicConv,
    "C3k2_DynamicConv": C3k2_DynamicConv,
    "RFAConv": RFAConv,
    "C3k2_RFAConv": C3k2_RFAConv,
}


def register_yolo11_custom_modules():
    for name, obj in _CUSTOM_SYMBOLS.items():
        setattr(tasks, name, obj)


def resolve_model_spec(model_spec: str) -> str:
    path = Path(model_spec)
    if path.is_file():
        return str(path.resolve())

    custom_path = CUSTOM_MODEL_DIR / path.name
    if custom_path.is_file():
        return str(custom_path)

    return model_spec


def create_yolo_model(model_spec: str) -> YOLO:
    register_yolo11_custom_modules()
    return YOLO(resolve_model_spec(model_spec))
