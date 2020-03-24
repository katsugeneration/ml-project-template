# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any
import importlib


def get_attribute(module_path: str) -> Any:
    """Return attribute from module path."""
    module_name = ".".join(module_path.split('.')[:-1])
    attribute_name = module_path.split('.')[-1]
    try:
        module = importlib.import_module(module_name)
    except Exception:
        module = get_attribute(module_name)
    return getattr(module, attribute_name)
