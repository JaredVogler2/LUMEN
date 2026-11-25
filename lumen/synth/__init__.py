"""Deterministic synthetic panel renderer (test fixtures and demo mode only)."""

from .render import RenderResult, SceneGeometry, project_points, render, scene_truth, srgb_decode, srgb_encode
from .scene import (
    BackgroundSpec,
    Box,
    CameraSpec,
    CurvedEdge,
    FloorSpec,
    LightSpec,
    PanelSpec,
    Placard,
    Scene,
    Seam,
    TagSticker,
    TrimSpec,
)

__all__ = [
    "BackgroundSpec", "Box", "CameraSpec", "CurvedEdge", "FloorSpec", "LightSpec", "PanelSpec",
    "Placard", "RenderResult", "Scene", "SceneGeometry", "Seam", "TagSticker", "TrimSpec",
    "project_points", "render", "scene_truth", "srgb_decode", "srgb_encode",
]
