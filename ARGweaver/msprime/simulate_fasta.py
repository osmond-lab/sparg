import msprime
import random

# Generate a random tree sequence with record_full_arg=True so that you get marked recombination nodes
ts_rs = random.randint(0,10000)   
ts = msprime.sim_ancestry(
    samples=2,
    recombination_rate=1e-8,
    sequence_length=3_000,
    population_size=10_000,
    record_full_arg=True,
    random_seed=ts_rs
)
mts_rs = random.randint(0,10000)
mts = msprime.sim_mutations(
    tree_sequence=ts, 
    rate=0.01,
    random_seed=mts_rs
)

mts.write_fasta("output.fa")


#for var in mts.variants():
#    print(var.site.position, var.alleles, var.genotypes, sep="\t")