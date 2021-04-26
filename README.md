

## Environment Setup

### Python and Jupyter

```
sudo pacman -S --needed python38
python3.8 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r python-requirements.txt
pip install -r jupyter-requirements.txt
jupyter contrib nbextension install
python -m jupyter notebook
```

### Poetry + Pyenv
This consider that both, `pyenv` and `poetry` are instaled:
```bash
pyenv install 3.8.0
pyenv local 3.8.0
poetry env use 3.8.0
poetry install
jupyter contrib nbextension install
# Run jupyter-lab or jupyter as you like:
poetry run jupyter-lab
poetry run jupyter notebook
```

### Dataset

First download the [Oxford TownCentre Dataset](https://academictorrents.com/details/35e83806d9362a57be736f370c821960eb2f2a01), place the mp4 video in this folder and run `sh convert_video.sh` from terminal.

