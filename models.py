"""
models.py: Core data models for Edge AI Dashcam
- Detection, Track, RiskEvent classes with serialization
"""
import json
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int

    def to_json(self):
        return json.dumps(asdict(self))

@dataclass
class Track:
    track_id: int
    bbox: List[int]
    conf: float
    class_id: int

    def to_json(self):
        return json.dumps(asdict(self))

@dataclass
class RiskEvent:
    event_type: str
    severity: str
    track_id: int
    ttc: float
    timestamp: float

    def to_json(self):
        return json.dumps(asdict(self))
