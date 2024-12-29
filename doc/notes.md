## Notes

Tried Gaussian Fourier features for proprio/action input but did not help.

adaLN(-Zero) for time conditioning seems to speed up training a bit initially, but does not make a significant difference after a while.

Overall, Gamma schedule in flow matching achieves better validation loss, but I usually seen linear schedule matching or even outperforming in validation accuracy with low threshold (i.e., predicted actions within small distance from the ground-truth ones).

Is flow matching / diffusion objective more stable than cross-entropy so larger lr can be used?
