import sparg
import msprime

ts = msprime.sim_ancestry(samples=2, record_full_arg=True)

print(ts)
