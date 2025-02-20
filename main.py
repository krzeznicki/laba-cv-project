import sys
from pathlib import Path
import importlib

# Get the absolute path of the FOOTBALLAI_ENDPROJECT root directory
ROOT_DIR = Path(__file__).resolve().parent  # FOOTBALLAI_ENDPROJECT
sys.path.append(str(ROOT_DIR))

import config.paths
importlib.reload(config.paths)
from config.paths import paths

import GUI.GUI
importlib.reload(GUI.GUI)
from GUI.GUI import GUI

if __name__ == '__main__':
    GUI()
