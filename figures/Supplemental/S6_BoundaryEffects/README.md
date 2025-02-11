# Figure S6 - Boundary Effects

Instructions for recreating:

1) Create the dataset by running `UniformStartInTwoDims.slim`. This will output the tree sequence file `rep0_S025_I1_R2_W300_D2.trees`.
2) Run `process.py` to estimate the location of ancestors within the ARG and generate the location error file `random_ancestors_1000_effective_dispersal.csv`.
3) Open `BoundaryEffects.ipynb` in Jupyter Lab to draw the plots shown in Supplementary Figure S5 of the manuscript.

Note: `random_ancestors_1000_estimated_dispersal.csv` is the location error file using the estimated rather than effective dispersal rate.