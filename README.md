# Instructions

## Download simulation outputs from Dryad

Place these in the simulations folder. The README.md within this folder explains the differences between the simulations.

## Create the conda environment from environment.yml and launch Jupyter Lab

The following commands will load all necessary dependencies including sparg from its GitHub repository as well as Jupyter Lab for viewing the figures. The conda environment is named `sparg`. The final command launches the Jupyter Lab notebook.

```
conda env create -f environment.yml
conda activate sparg
jupyter lab
```

## Recreate Figures

Within the figures folder, there is a folder for each figure in the manuscript. Figures that are constructed in Python have associated Jupyter Notebooks with instructions for recreating the plots.








# -----------------
# python set up
# -----------------

# install python3
# install jupyter-lab 

# set up virtual env for python
python3 -m venv sparg-arg

# start virtual env
source sparg-arg/bin/activate

# make venv available in jupyter
pip install ipykernel 
python -m ipykernel install --user --name=sparg-arg

# will need to pip install some packages (and dependencies for argutils, not listed here -- see their requirements.txt file) to run notebooks
pip install pyslim==1.0b1
pip install tskit
pip install matplotlib

# got argutils from git clone https://github.com/tskit-dev/what-is-an-arg-paper (I've put this in the jupyter-notebooks folder to easily load from notebooks)

#--------------
# run simulation
#-------------

# get SLiM
mkdir programs
cd programs
git clone https://github.com/MesserLab/SLiM.git
cd SLiM
mkdir build
cd build
cmake ../
make slim

# run SLiM simulation
mkdir slim/output
programs/SLiM/build/slim -d L=1e6 -d r=1e-8 -d R=2 -d SIGMA_int=1 -d SIGMA_disp=1 -d K=1 -d W=10.0 -d t=1e4 -d N0=10 -d "filename='slim/output/simple_space.trees'" slim/simple_space.slim 

#-----------------
# analyze 
# --------------

# open jupyter
jupyter-lab
# and from there navigate to to the figure of interest in the figures/
#For figures 4 and 5, the .ipynb file required to generate the data files and the figures are found inside the folder.
#For figures 2,3,6 and 7 the corresponding figure folder contains a folder called Code which has the .ipynb file 
#For figure 2, the code to generate the data file is also provided as a python file so it can be run on a cluster. 

 

