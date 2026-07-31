"""bssfp
"""
import copy

import matplotlib.pyplot
import mrsd

# Define sequence parameters (arbitrary units): echo and repetition times,
# durations of ramp, pulses, encoding and readout
TE, TR = 5, 10
d_ramp, d_pulse,  d_readout = 0.1, 1, 1

# Create the underlying Matplotlib objects and the diagram
figure, plot = matplotlib.pyplot.subplots(tight_layout=True)
diagram = mrsd.Diagram(
    plot, ["RF", "$G_{readout}$", "Signal"])

# Slice-selective pulse of the first TR
excitation= diagram.rf_pulse(
    "RF", d_pulse, 0.2, center=0)

#readout
readout = diagram.gradient("$G_{readout}$", d_readout, 0.5, d_ramp, center=TE)

#dephasing
dephasing = diagram.gradient("$G_{readout}$", d_readout/2, -0.5, d_ramp, end=readout.begin)
  
#Spoiler
spoiler = diagram.gradient("$G_{readout}$", d_readout/2, -0.5, d_ramp, begin=readout.end)

#echo
echo=diagram.echo("Signal",d_readout,1,center=TE)

# Start of next TR
diagram.add("RF", copy.copy(excitation).move(TR))

# Add annotations: flip angles and TE/TR intervals
diagram.annotate("RF", 0.2, 1, r"$\alpha$")
diagram.annotate("RF", TR+0.2, 1, r"$\alpha$")
diagram.interval(0, TE, -1.5, "TE")
diagram.interval(0, TR, -2.5, "TR")

matplotlib.pyplot.show()
# %%
