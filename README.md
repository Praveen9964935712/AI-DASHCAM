# Edge AI Dashcam

A real-time, privacy-first, explainable AI dashcam for Indian roads.

## Features
- Real-time object detection (YOLOv8)
- Multi-object tracking
- Risk scoring (TTC, headway, lane deviation)
- Privacy filtering (license plate blurring)
- Incident logging and hash-signed packs
- Modular, production-grade codebase

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `python tests.py`
3. Run main pipeline: `python main.py`

## Modules
- `detection.py`: Object detection
- `tracking.py`: Multi-object tracking
- `risk.py`: Risk scoring
- `privacy.py`: Privacy filtering
- `incident.py`: Incident logging
- `models.py`: Data models
- `tests.py`: Unit tests

## Roadmap
- Lane detection
- Explainable AI overlays
- Insurance-grade incident packs
- Mobile/web dashboard
