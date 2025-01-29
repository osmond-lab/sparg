# Instructions

## Create the conda environment from environment.yml and launch Jupyter Lab

The following commands will load all necessary dependencies including sparg from its GitHub repository as well as Jupyter Lab for viewing the figures. The conda environment is named `sparg`. The final command launches the Jupyter Lab notebook.

```
conda env create -f environment.yml
conda activate sparg
jupyter lab
```

## Recreate Figures

Within the figures folder, there is a folder for each figure in the manuscript. There is a README within each folder with instructions for recreating the dataset used in that analysis. Figures that are constructed in Python have associated Jupyter Notebooks with instructions for recreating the plots.