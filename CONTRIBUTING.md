# Contributing to HAT-Light Satellite Super-Resolution

Thank you for your interest in contributing to this open-source project! We welcome contributions from students, researchers, and geospatial developers.

## Ways to Contribute

1. **Bug Reports & Feature Requests**: Open an issue describing the problem or requested enhancement.
2. **New Sensor Bands**: Extend the dataset ingestion pipeline to support additional Sentinel-2 bands (e.g., SWIR B11/B12 or Red Edge B05/B06/B07) or Landsat-8/9.
3. **Model Improvements**: Experiment with attention window sizes, lightweight transformer backbones, or quantization (INT8/FP16).
4. **Studio Enhancements**: Contribute features to the React frontend or FastAPI backend.

## Development Workflow

1. Fork and clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Test changes locally by launching `python app.py`.
5. Submit a Pull Request with a clear description of your changes.

## Code Style

- Follow PEP 8 guidelines for Python code.
- Keep functions modular and include docstrings.
