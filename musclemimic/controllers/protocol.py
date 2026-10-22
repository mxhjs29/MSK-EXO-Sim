"""Host-side controllers return explicit commands; no neural inference inside physics."""
from typing import Protocol, Mapping, Any


class ExternalController(Protocol):
    def reset(self, seed: int) -> None: ...
    def command(self, context: Mapping[str, Any]) -> Any: ...
