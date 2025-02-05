# Instructions

## Create the conda environment from environment.yml and launch Jupyter Lab

The following commands will load all necessary dependencies including sparg from its GitHub repository as well as Jupyter Lab for viewing the figures. The conda environment is named `sparg`. The final command launches the Jupyter Lab notebook.

```
conda env create -f environment.yml
conda activate sparg
jupyter lab
```

## Alternately, use virtualenv

An alternative to conda is to use virtualenv. First create and load a virtual environment, e.g.,

```
python3 -m venv ~/.virtualenvs/sparg #create env
source ~/.virtualenvs/sparg/bin/activate #activate env
```

then load the packages and launch jupyter lab,

```
pip install -r requirements.txt #load required packages
ipython kernel install --user --name=sparg #make env available on jupyter
jupyter-lab #launch jupyter
```

Open notebook of choice (see below) and load sparg kernel.

## Recreate Figures

Within the figures folder, there is a folder for each figure in the manuscript. There is a README within each folder with instructions for recreating the dataset used in that analysis. Figures that are constructed in Python have associated Jupyter Notebooks with instructions for recreating the plots.
