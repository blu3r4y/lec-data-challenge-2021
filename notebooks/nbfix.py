import logging
import os
import sys
from warnings import filterwarnings

from kedro.framework.session import KedroSession
from kedro.framework.session.session import _activate_session
from kedro.framework.startup import _get_project_metadata

log = logging.getLogger(__name__)


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


def filter_warnings():
    """
    Do not show me some odd warnings about `should_run_async` in notebooks
    """
    filterwarnings("ignore", ".*`should_run_async`.*")


def load_catalog():
    """
    Loads the kedro project and returns the catalog objects
    """

    # assume that the project is in the current directory
    project_path = os.path.abspath(".")

    metadata = _get_project_metadata(project_path)
    session = KedroSession.create(metadata.package_name, project_path)
    _activate_session(session, force=True)

    context = session.load_context()
    catalog = context.catalog

    log.info("Loaded Kedro project %s", str(metadata.project_name))

    return catalog


# fix the path immediately upon import
add_import_path()
fix_root_path()

# filter some warnings in notebooks
filter_warnings()
