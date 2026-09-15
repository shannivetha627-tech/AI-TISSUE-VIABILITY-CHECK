import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_session_secret_is_read_from_env():
    os.environ["SESSION_SECRET"] = "stable-session-secret"
    os.environ.pop("TISSUE_VIABILITY_SECRET_KEY", None)
    import app as app_module
    importlib.reload(app_module)
    assert app_module.app.config["SECRET_KEY"] == "stable-session-secret"
