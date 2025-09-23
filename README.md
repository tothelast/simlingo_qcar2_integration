# SimLingo-Qcar2 Integration

This project integrates the SimLingo Vision-Language-Action (VLA) model with the Qcar2 platform for autonomous driving in the Qlabs virtual environment.

## Overview

SimLingo is a state-of-the-art VLA model originally developed for the CARLA simulator. This integration adapts it to work with Quanser's Qcar2 vehicle in the Qlabs virtual environment, enabling:

- Vision-based autonomous driving using front camera input
- Language capabilities (VQA, commentary, instruction following)
- Real-time control of Qcar2 vehicle through QVL SDK

## Project Structure

```
simlingo_qcar2_integration/
├── src/                    # Source code
│   ├── adapters/          # Data and control adapters
│   ├── models/            # Model loading and inference
│   └── integration/       # Main integration logic
├── models/                # Downloaded model files
├── tests/                 # Test scripts
├── docs/                  # Documentation
├── config/                # Configuration files
└── requirements.txt       # Python dependencies
```

## Key Components

### Data Adapter
- Converts Qcar2 camera data (820x410 or 640x480) to SimLingo format (1024x512)
- Handles image preprocessing and normalization

### Control Adapter  
- Maps SimLingo action outputs to Qcar2 control commands
- Interfaces with `set_velocity_and_request_state()` method

### Integration Bridge
- Orchestrates the complete pipeline from sensor input to vehicle control
- Manages model inference and real-time operation

## Requirements

- Ubuntu 24.04
- Python 3.8+
- PyTorch 2.2.0
- Qlabs virtual environment
- Qcar2 SDK (QVL)
- HuggingFace transformers

## Installation

1. Set up Python virtual environment:
```bash
python -m venv simlingo_env
source simlingo_env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download SimLingo model (requires HuggingFace token)

## Usage

1. Launch Qlabs virtual environment
2. Run the integration script:
```bash
python src/main.py
```

## Adaptations from CARLA to Qcar2

- **Image Format**: Resizing from Qcar2's camera resolution to SimLingo's expected input
- **Control Interface**: Mapping from SimLingo's action space to Qcar2's control API
- **Environment**: Bridging CARLA simulator interface to Qlabs/Qcar2 interface

## Testing

Run tests to validate the integration:
```bash
python -m pytest tests/
```

## Documentation

See `docs/` folder for detailed documentation on:
- Integration architecture
- API reference
- Troubleshooting guide
