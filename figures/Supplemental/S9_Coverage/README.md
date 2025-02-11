# Figure S9 - Coverage

Instructions for recreating:

1) Create the dataset by running `UniformStartInTwoDims.slim`. This will output the tree sequence file `rep5_S025_I1_R2_W100_D2.trees`.
2) Run `process.py` to estimate the location of ancestors within the ARG and generate the location error file `random_ancestors_1000_effective_dispersal.csv`.
3) Open `Coverage.ipynb` in Jupyter Lab to draw the plots shown in Supplementary Figure S9 of the manuscript.