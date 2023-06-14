import tskit
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/tskit_arg_visualizer/visualizer")
import visualizer


ts = tskit.load("condensed.trees")
d3arg = visualizer.D3ARG(ts=ts)
d3arg.draw(width=2500, height=2500, line_type="line", y_axis_scale="log_time")