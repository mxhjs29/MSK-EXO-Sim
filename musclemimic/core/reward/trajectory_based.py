"""Compatibility entry point; implementation: musclemimic.rewards.common.imitation."""
import importlib
import sys
if __name__ == "__main__":
    import runpy
    runpy.run_module("musclemimic.rewards.common.imitation", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("musclemimic.rewards.common.imitation")
