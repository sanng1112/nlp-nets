import importlib
import os
from pathlib import Path
from typing import Sequence

from utils import logger

LIBRARY_ROOT = Path(__file__).resolve().parent.parent


def import_modules_from_folder(
    folder_name: str, extra_roots: Sequence[str] = ()
) -> None:
    """
    Auto-import all Python modules from a folder (and sub-folders)
    to trigger registry decorators.

    Skips files starting with '.' or '_' (except __init__.py).
    """
    if not LIBRARY_ROOT.joinpath(folder_name).exists():
        logger.error(f"{folder_name} doesn't exist in the nlp-nets library root directory.")

    for base_dir in [".", *extra_roots]:
        pattern = os.path.join(base_dir, folder_name, "**/*.py")
        for path in LIBRARY_ROOT.glob(pattern):
            filename = path.name
            if filename[0] not in (".", "_") or filename == "__init__.py":
                module_name = str(
                    path.relative_to(LIBRARY_ROOT).with_suffix("")
                ).replace(os.sep, ".")
                importlib.import_module(module_name)
