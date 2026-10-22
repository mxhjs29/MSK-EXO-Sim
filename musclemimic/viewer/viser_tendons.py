"""Compatibility entry point; implementation: musclemimic.visualization.viewer.viser_tendons."""
import importlib
import sys
if __name__ == "__main__":
    import runpy
    runpy.run_module("musclemimic.visualization.viewer.viser_tendons", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("musclemimic.visualization.viewer.viser_tendons")
