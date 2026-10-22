"""Compatibility entry point; implementation: musclemimic.runner.jobs.payload_background."""
import importlib
import sys
if __name__ == "__main__":
    import runpy
    runpy.run_module("musclemimic.runner.jobs.payload_background", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("musclemimic.runner.jobs.payload_background")
