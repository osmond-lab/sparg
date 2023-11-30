import numpy as np
import matplotlib.pyplot as plt

bench = np.loadtxt("benchmarking.csv", delimiter=",")

plt.scatter(bench[5], bench[7], label="paths")
plt.scatter(bench[5], bench[8], label="bottom up")
plt.scatter(bench[5], bench[9], label="minimal")

plt.xlim(0,1000)
plt.legend()
plt.show()