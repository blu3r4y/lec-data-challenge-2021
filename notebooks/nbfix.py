import os
import sys


def add_import_path():
    """
    A dirty hack that tries to append the current and parent
    folder to the path so that one can do `import ldc2021`
    """

    src_root = os.path.abspath(os.path.join(".", "src"))
    src_parent = os.path.abspath(os.path.join("..", "src"))
    if src_root not in sys.path:
        sys.path.append(src_root)
    if src_parent not in sys.path:
        sys.path.append(src_parent)


def fix_root_path():
    """
    Move to the root directory so that data from the catalog can
    be loaded (because it will load stuff relative from there)
    """

    if "notebooks" in os.getcwd():
        os.chdir(os.path.join(os.getcwd(), ".."))


# fix the path immediately upon import
add_import_path()
fix_root_path()
