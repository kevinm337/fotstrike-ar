from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: Literal["player", "ball"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., min_length=4, max_length=4)  # [x1,y1,x2,y2]
