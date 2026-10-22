"""Recorded quantities carry physical units and sampling rate explicitly."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RecordingSchema:
    frequency_hz: float
    joint_names: tuple[str, ...]
    muscle_names: tuple[str, ...]
    contact_names: tuple[str, ...] = ()
    torque_unit: str = "Nm"
    contact_unit: str = "N"
    activation_quantity: str = "activation"

    def __post_init__(self):
        if self.frequency_hz <= 0:
            raise ValueError("Positive sampling frequency required")
