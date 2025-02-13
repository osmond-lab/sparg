# Figure S10 - Time Stamped Location Error

Instructions for recreating:

1) Create the dataset by running `UniformStartInTwoDims.slim`. This will output the tree sequence file `rep5_S025_I1_R2_W100_D2.trees`.
2) Run `process.py` to estimate the location of ancestors within the ARG and generate the location error file `random_ancestors_1000_effective_dispersal.csv`.
3) Open `FigS10.ipynb` in Jupyter Lab to draw the plots shown in Figure S10 of the manuscript.