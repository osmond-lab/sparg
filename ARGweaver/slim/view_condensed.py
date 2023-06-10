import tskit
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/tskit_arg_visualizer/visualizer")
import visualizer


ts = tskit.load("condensed.trees")
d3arg = visualizer.D3ARG(ts=ts)
d3arg.draw(width=1000, height=1500, line_type="ortho", y_axis_scale="rank")