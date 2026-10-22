"""Compatibility entry point; implementation: musclemimic.runner.workflows.prepare_walk."""
import importlib
import sys
if __name__ == "__main__":
    import runpy
    runpy.run_module("musclemimic.runner.workflows.prepare_walk", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("musclemimic.runner.workflows.prepare_walk")
