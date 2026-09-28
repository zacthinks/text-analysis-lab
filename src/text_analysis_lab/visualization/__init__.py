"""Artifact-bound, corpus-scalable visualization helpers."""

from text_analysis_lab.visualization.accessor import ArtifactVisualization
from text_analysis_lab.visualization.histogram import HistogramResult, histogram
from text_analysis_lab.visualization.scatter import scatter

__all__ = ["ArtifactVisualization", "HistogramResult", "histogram", "scatter"]
