# Figure 6

Instructions for recreating:

1) Create the dataset by running `UniformStartInTwoDims.slim`. This will output the tree sequence file `rep5_S025_I1_R2_W100_D2.trees`.
2) Run `process.py` to estimate the location of ancestors within the ARG and generate the location error file `random_ancestors_1000_effective_dispersal.csv`.
3) Open `Fig6.ipynb` in Jupyter Lab to draw the plots shown in Figure 6 of the manuscript.

Note: `random_ancestors_1000_estimated_dispersal.csv` is the location error file using the estimated rather than effective dispersal rate.