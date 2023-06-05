import msprime
import random

# Generate a random tree sequence with record_full_arg=True so that you get marked recombination nodes
ts_rs = random.randint(1,10000)   
ts = msprime.sim_ancestry(
    samples=100,
    recombination_rate=1.5e-8,
    sequence_length=10_000,
    population_size=10_000,
    record_full_arg=True,
    random_seed=ts_rs
)

mts_rs = random.randint(0,10000)
mts = msprime.sim_mutations(
    tree_sequence=ts, 
    rate=2.5e-8,
    random_seed=mts_rs
)

print(ts_rs)
print(mts_rs)
print(ts.num_trees)

mts.write_fasta("run6/"+str(ts_rs)+"_"+str(mts_rs)+".fa")


#for var in mts.variants():
#    print(var.site.position, var.alleles, var.genotypes, sep="\t")