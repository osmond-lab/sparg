for s_disp in $(seq 0.75 0.25 1)
do
for rep in {1..10}
do 
slim -d SIGMA_disp=$s_disp -d rep=$rep simple_space.slim;
echo $rep
done
done
