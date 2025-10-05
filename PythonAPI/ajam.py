import importlib
import sys

def get_version(pkg_name):
    try:
        module = importlib.import_module(pkg_name)
        version = getattr(module, "__version__", None)
        return version or "built-in"
    except Exception:
        return "not installed"

libs = [
    "earthaccess",
    "xarray",
    "numpy",
    "flask",
    "math",
    "os",
    "json",
    "datetime",
    "pathlib",
    "threading",
]

for lib in libs:
    print(get_version(lib))