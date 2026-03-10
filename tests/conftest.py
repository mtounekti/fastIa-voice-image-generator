import sys
import os

# Ajoute la racine du projet au chemin Python
# pour que pytest trouve api.py depuis le dossier tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))