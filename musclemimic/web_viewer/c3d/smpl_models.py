"""Compatibility entry point; implementation: musclemimic.visualization.web_viewer.c3d.smpl_models."""
import importlib
import sys
if __name__ == "__main__":
    import runpy
    runpy.run_module("musclemimic.visualization.web_viewer.c3d.smpl_models", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("musclemimic.visualization.web_viewer.c3d.smpl_models")
