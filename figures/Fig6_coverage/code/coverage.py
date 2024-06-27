import pandas
import matplotlib.pyplot as plt
import scipy.stats as st

expected_zs = []
expected_percent_within = []
for i in range(0,300):
    value = st.norm.cdf(i/100)
    expected_zs.append(i/100)
    expected_percent_within.append(value - (1-value))


coverage = pandas.read_csv("coverage_labmeeting_today.csv")


plt.plot(expected_percent_within, expected_percent_within, color="gray", linestyle='dashed')

step = 10000
for c in range(0,40000,step):#["epoch_low"].unique():
    percent_within = []
    in_epoch = coverage[(coverage["epoch_low"]>c) & (coverage["epoch_low"]<=c+step)]
    for i in coverage["num_std"].unique():
        filtered = in_epoch[in_epoch["num_std"]==i]
        percent_within.append(filtered["within"].sum()/filtered["total"].sum())
    if len(in_epoch.index) > 0:
        plt.plot(expected_percent_within, percent_within, label=str(c) + " - " + str(c+step))

plt.legend()
plt.xlabel("Expected")
plt.ylabel("Observed")
plt.show()