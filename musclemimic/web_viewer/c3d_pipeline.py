"""Compatibility entry point; implementation: musclemimic.visualization.web_viewer.c3d_pipeline."""
import importlib
import sys
if __name__ == "__main__":
    import runpy
    runpy.run_module("musclemimic.visualization.web_viewer.c3d_pipeline", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("musclemimic.visualization.web_viewer.c3d_pipeline")
