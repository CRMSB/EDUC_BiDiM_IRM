""" Draw the sequence diagram of a multi-echo gradient echo sequence, with
    bipolar readout gradients.
"""

import copy

import matplotlib.pyplot
import mrsd
import numpy

# Define the tissue and sequence parameters (arbitrary units): T2,
# echo and repetition times, durations of ramp, pulses, encoding and readout,
# length of the echo train
T2 = 60
TE, TR = 10, 1000
d_pulse, d_crusher, d_ramp ,d_readout= 1, 1, 0.1,1
train_length = 6

# Create the underlying Matplotlib objects and the diagram
figure, plot = matplotlib.pyplot.subplots(tight_layout=True, figsize=(10,5))
diagram = mrsd.Diagram(plot, ["RF","$G_z$","Echoes"])

# Add the excitation RF pulse, show the beginning of the next repetition
excitation = diagram.rf_pulse(
    "RF", d_pulse,0.25,center=0)
diagram.annotate("RF", excitation.center+0.2, 1, "90°")

diagram.interval(0, TE/2, -1, "TE/2")

diagram.interval(0, TE, -1.5, "TE")

# Add the other echoes in the train
ind=0
for echo in range(1, train_length):
    # Add the refocalization RF pulse
    refocalization = diagram.rf_pulse("RF", d_pulse, 0.5, center=ind*TE+TE/2)
    diagram.annotate("RF", refocalization.center+0.2, 1, "180°")
    # Add the crusher
    crusher1 = diagram.gradient("$G_z$", d_crusher, 1, d_ramp, end=refocalization.begin)
    # Add the crusher
    crusher2 = diagram.gradient("$G_z$", d_crusher, 1, d_ramp, begin=refocalization.end)
    #Add echo
    echo_amplitude=numpy.exp(-(ind+1)*TE/T2)
    echo=diagram.echo("Echoes",d_readout,echo_amplitude,center=ind*TE+TE)
    ind=ind+1
matplotlib.pyplot.show()