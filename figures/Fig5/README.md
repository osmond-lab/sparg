# Figure 5

Instructions for recreating:

1) Open `Fig5_DispersalRate.ipynb` in Jupyter Lab to draw the plots shown in Figure 5 of the manuscript. This includes all of the instructions for each subpanel.
2) The .trees files used to generate the figures can be downloaded from here - https://drive.google.com/drive/folders/1u1Y85fALAqLyY6f9yb-EP6Lk8HRH2nsR?usp=sharing 
3) If you wish to generate the .trees files yourself, you will need to run the .slim file (following the instructions in the jupyter-notebook) on a cluster. You can install SLiM on the cluster by running the following lines on the cluster terminal. Further, you might have to change the directory for SLiM in the SLiM_Parallelize.sh file. 

wget https://github.com/MesserLab/SLiM/releases/download/v4.0.1/SLiM.zip -P {PROGRAMDIR} <br>
cd {PROGRAMDIR} <br>
unzip SLiM.zip <br>
rm SLiM.zip <br>
module load cmake/3.21.4<br>
module load gcc/8.3.0<br>
mkdir build<br>
cd build<br>
cmake ../SLiM<br>
make slim<br>
cd .. <br>
mv build/ SLiM_build
