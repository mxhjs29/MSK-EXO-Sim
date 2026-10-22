"""Compatibility for the independent muscle recruitment controller."""
import sys
from musclemimic.controllers.muscle_recruitment import frozen_policy
sys.modules[__name__] = frozen_policy
