import click
import argutils
import tskit
import matplotlib.pyplot as plt


@click.group()
def cli():
    pass


@click.command()
@click.argument("num_samples", type=int)
@click.argument("output", type=click.File("wb"))
@click.option("--rho", "-r", type=float, default=0)
@click.option("--sequence-length", "-L", type=int, default=10)
@click.option("--seed", "-s", type=int, default=42)
def simulate(num_samples, sequence_length, rho, seed, output):
    """
    Simulate an ARG under the coalescent with recombination and write
    out the explicitly resolved ancestry result to file.
    """
    ts = argutils.sim_coalescent(num_samples, rho, sequence_length, seed=seed)
    print(ts)
    ts.dump(output)


@click.command()
@click.argument("input", type=click.File("rb"))
@click.argument("output", type=click.Path())
def draw(input, output):
    ts = tskit.load(input)
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 6))
    argutils.draw(ts, ax1, draw_edge_widths=True)
    plt.savefig(output)


cli.add_command(simulate)
cli.add_command(draw)

if __name__ == "__main__":
    cli()
