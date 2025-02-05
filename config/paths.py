import os
from pathlib import Path

# Get the absolute path of the FOOTBALLAI_ENDPROJECT root directory
HOME = Path(__file__).resolve().parent.parent  # Moves up to FOOTBALLAI_ENDPROJECT

# Define paths
DATA = HOME / 'data'  # Path where the data sets will be saved
MODELS = HOME / 'Models'  # The path in wich the models are located

# Store paths in a dictionary
paths = {
    'HOME': HOME,
    'DATA': DATA,
    'MODELS': MODELS
}

for path in paths.values():
    path.mkdir(parents=True, exist_ok=True)
