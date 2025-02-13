# Figure 5

Instructions for recreating:

1) Open `Fig5_DispersalRate.ipynb` in Jupyter Lab to draw the plots shown in Figure 5 of the manuscript. This includes all of the instructions for each subpanel.
2) To run the .slim file on a cluster you will have to install SLiM on the cluster. This can be done by running the following lines on the cluster terminal
    wget https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM.zip -P {PROGRAMDIR}
    cd {PROGRAMDIR}
    unzip SLiM.zip
    rm SLiM.zip
    module load cmake/3.21.4
    module load gcc/8.3.0
    mkdir build
    cd build
    cmake ../SLiM
    make slim
    cd .. 
    mv build/ SLiM_build
