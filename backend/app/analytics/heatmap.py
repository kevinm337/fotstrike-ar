from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class HeatmapGrid:
    """
    Stores counts on a coarse grid (grid_y x grid_x).
    Can render to a colored overlay on top of a video frame.
    """

    width: int
    height: int
    grid_x: int
    grid_y: int

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        self.grid_x = int(self.grid_x)
        self.grid_y = int(self.grid_y)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("HeatmapGrid width/height must be > 0")
        if self.grid_x <= 0 or self.grid_y <= 0:
            raise ValueError("HeatmapGrid grid_x/grid_y must be > 0")

        self.grid = np.zeros((self.grid_y, self.grid_x), dtype=np.int32)

        # cell sizes in pixels
        self._cell_w = max(1.0, self.width / float(self.grid_x))
        self._cell_h = max(1.0, self.height / float(self.grid_y))

    def add_position(self, x: float, y: float, amount: int = 1) -> None:
        """
        Convert (x,y) pixel coords into grid indices and increment.
        """
        # clamp into frame bounds
        x = float(max(0.0, min(x, self.width - 1.0)))
        y = float(max(0.0, min(y, self.height - 1.0)))

        gx = int(x / self._cell_w)
        gy = int(y / self._cell_h)

        gx = max(0, min(self.grid_x - 1, gx))
        gy = max(0, min(self.grid_y - 1, gy))

        self.grid[gy, gx] += int(amount)

    def max_value(self) -> int:
        return int(self.grid.max()) if self.grid.size else 0

    def to_uint8(self, clip_max: Optional[int] = None) -> np.ndarray:
        """
        Normalize grid -> uint8 image (0..255).
        """
        g = self.grid.astype(np.float32)

        m = float(g.max()) if clip_max is None else float(clip_max)
        if m <= 0.0:
            return np.zeros_like(self.grid, dtype=np.uint8)

        g = (g / m) * 255.0
        return np.clip(g, 0, 255).astype(np.uint8)

    def render_colormap(
        self,
        colormap: int = cv2.COLORMAP_JET,
        blur_ksize: int = 0,
    ) -> np.ndarray:
        """
        Returns a BGR heatmap image sized to (height, width).
        """
        u8 = self.to_uint8()

        # Upscale from grid size -> video size
        heat_small = cv2.resize(u8, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        if blur_ksize and blur_ksize > 1:
            # make it smoother (must be odd)
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            heat_small = cv2.GaussianBlur(heat_small, (blur_ksize, blur_ksize), 0)

        heat_bgr = cv2.applyColorMap(heat_small, colormap)
        return heat_bgr

    def overlay_on_frame(
        self,
        frame_bgr: np.ndarray,
        alpha: float = 0.35,
        colormap: int = cv2.COLORMAP_JET,
        blur_ksize: int = 0,
    ) -> np.ndarray:
        """
        Blend heatmap onto a frame.
        alpha=0 -> original frame, alpha=1 -> full heatmap
        """
        alpha = float(max(0.0, min(1.0, alpha)))

        heat_bgr = self.render_colormap(colormap=colormap, blur_ksize=blur_ksize)

        if heat_bgr.shape[:2] != frame_bgr.shape[:2]:
            heat_bgr = cv2.resize(heat_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))

        out = cv2.addWeighted(frame_bgr, 1.0 - alpha, heat_bgr, alpha, 0.0)
        return out

    def save_png(
        self,
        path: str | Path,
        colormap: int = cv2.COLORMAP_JET,
        blur_ksize: int = 0,
    ) -> Path:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        img = self.render_colormap(colormap=colormap, blur_ksize=blur_ksize)
        cv2.imwrite(str(p), img)
        return p
