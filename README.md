

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

### Dataset Download

[Oxford TownCentre Dataset](https://academictorrents.com/details/35e83806d9362a57be736f370c821960eb2f2a01)
