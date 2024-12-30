## Notes

Tried Gaussian Fourier features for proprio/action input but did not help.

adaLN(-Zero) for time conditioning seems to speed up training a bit initially, but does not make a significant difference after a while.

Overall, Gamma schedule in flow matching achieves better validation loss, but I usually see linear schedule matching or even outperforming in validation accuracy with low threshold (e.g.,, predicted actions within 0.05 from the normalized ground-truth ones in all dimensions).

Is flow matching / diffusion objective more stable than cross-entropy so larger lr can be used?
