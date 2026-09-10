"""Optional integrations for Text Analysis Lab."""

from text_analysis_lab.integrations.geco import (
    GeCoIntegrationError,
    GeCoManager,
    GeCoPredictorRef,
    LinkedGeCoWorkspace,
    TeALGeCoProvider,
)

__all__ = [
    "GeCoIntegrationError",
    "GeCoManager",
    "GeCoPredictorRef",
    "LinkedGeCoWorkspace",
    "TeALGeCoProvider",
]
