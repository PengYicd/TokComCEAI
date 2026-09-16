# Token channel

`tokchan.py` applies a codebook-level token channel. At each supported SNR, tokens are independently erased to `[mask]` according to the TER column in `snr_ber_ter.csv`.

`snr_to_ber_ter_sim.m` is the MATLAB simulation used to generate that mapping. The paper experiments use SNR values `{0, 1, 2, 3, 4}` dB.
