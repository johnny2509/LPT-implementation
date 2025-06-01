LPT_implementation.py: 

- Obtaining the power spectrum from CLASS
- Determination of growth factors
- Determination of displacement fields for 1LPT and 2LPT
- Plotting of power spectra for 1LPT and 2LPT for different redshifts
- Plotting histograms of shell crossing for different redshifts
- Visualization of density contrast 2D-slices for 1LPT and 2LPT
- Documentation for figures used in the thesis

2LPT_delta_generator.py:
Modified code of Simon Schmidt Thomsen, with added parts of LPT_implementation.py to generate 100 2LPT density contrast fields with corresponding A_s values.

loss-val-loss-predictions-z0-z10:
Data the neural network yields for both z=0 and z=10 in form of npy files for loss and validation loss, and txt files for A_s.

network_A_s.py:
Plotting of the data i folder "loss-val-loss-predictions-z0-z10". 
