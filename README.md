# Realtime Filter MVP (Python / PyQt6 / pyqtgraph)

Realtime demo app (MVP) showing input wave, filter responses and filtered output.
Includes PyInstaller spec and GitHub Actions to build a Windows executable.

## Run locally (dev)
1. Create virtualenv or use conda
2. `pip install -r requirements.txt`
3. `python src/realtime_filter_mvp.py`

## Build Windows executable (local)
Run the included script:
```powershell
.\build_exe.bat
