# Project Context

## Overview
- Pet behavior monitoring app built from raw IMU sensor windows.
- Current notebook is renamed to `pet_behavior_monitoring_app.ipynb` and covers data prep, windowing, model training, evaluation, and explainability.

## Architecture
- Backend structure: not extracted yet; notebook currently owns preprocessing and inference logic.
- Frontend structure: not started yet.
- Model integration: 1D CNN + BiLSTM on IMU channels with ODBA features.

## Features Implemented
- [x] Data ingestion and cleaning
- [x] Windowing and dog-level split
- [x] Model training and evaluation
- [x] Basic explainability plots
- [ ] API wrapper
- [ ] Frontend app
- [ ] History tracking

## Current Task
- Notebook framing has been shifted toward a product build; next work is wiring the app layer to exported model artifacts.

## Next Steps
- Extract preprocessing and model code into reusable Python modules.
- Add a small inference API around the trained model.
- Create a simple upload-and-predict UI.
- Export a real checkpoint metadata bundle from the notebook so the API can load it without manual setup.

## Decisions Made
- Keep the first build single-label and easy to validate.
- Prefer dog-level splits to avoid identity leakage.

## Known Issues / Limitations
- The notebook still contains some research-heavy narrative and citations.
- The app layer has not been created yet.

## Setup Instructions (Short)
- Open the notebook in Jupyter or VS Code.
- Run the cells from top to bottom after ensuring the Python environment has the required packages.

## Notes for Future Contributors
- Keep changes modular and testable.
- Update this file after meaningful changes so the next person can resume quickly.
- Work is currently on git branch `feature/app-layer`.