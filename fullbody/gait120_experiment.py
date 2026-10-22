"""Compatibility entry point; implementation: musclemimic.cli.train_gait120."""
import importlib
import sys
if __name__ == "__main__":
    import runpy
    runpy.run_module("musclemimic.cli.train_gait120", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("musclemimic.cli.train_gait120")
