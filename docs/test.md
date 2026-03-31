Column = position 
Row = channel 

Kernel = size of the window (3). Correct'
Filter = one set of trained weights (the numbers we learned), not “trained data.'

mean and variance are per channel, not per position.


normalized = normalized = (value − mean) / √(variance + ε) 
ε is a constant, usually something like 0.00001 (or 1e-5).

After the multiplication (Conv1d):

BN – use mean and variance (per channel), then for every value in that channel apply:
normalized = (value − mean) / √(variance + ε)
(ε is a small constant, e.g. 0.00001, so we never divide by zero.)

ReLU – then apply ReLU (negative → 0).

So: multiplication (conv) → BN (mean, variance, that formula for all values) → ReLU.

Purpose
BN keeps the output in a normal range (centered, stable scale) so values don’t keep growing (or shrinking) as we stack more layers.
ReLU adds non-linearity (and zeros out negatives).


Dropout randomly sets some values to 0 so the network doesn’t depend on a few specific numbers and learns more robust patterns.


Squeeze - average of the rows 
Excitation - small network (two layers) turns those numbers into weights between 0 and 1.
Scale - multiply each channel by its weight (important channels stay strong).