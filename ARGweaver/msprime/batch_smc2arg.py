import glob
import os

smc = glob.glob("run4/ARGweaver_output/smc/*")

print(smc)

for arg in smc:
    split_path = arg.split("/")
    split_filename = split_path[-1].split(".")
    output = "/".join([split_path[0], split_path[1], "arg", ".".join([split_filename[0], split_filename[1], "arg"])])
    print(arg, output)
    os.system(" ".join(["smc2arg", arg, output]))