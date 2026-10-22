"""Compatibility entry point; implementation: musclemimic.scenes.composition."""
import importlib
import sys
if __name__ == "__main__":
    import runpy
    runpy.run_module("musclemimic.scenes.composition", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("musclemimic.scenes.composition")
