'''In this .py file, the Loss and Validation loss
curves are plotted together with the true value for
A_s vs the Neural Network's predictions of A_s. 
GitHub Copilot Chat was used to generate blocks of 
code. To run the below code download the six files 
in the folder "loss-val-loss-predictions-z0-z10".'''

#%%

import numpy as np
import matplotlib.pyplot as plt

'''loading files and skipping the first row'''
loss_file_path_1 = "loss-val-loss-predictions-z0-z10/loss-z0-Johnny.npy"
val_loss_file_path_1 = "loss-val-loss-predictions-z0-z10/val-loss-z0-Johnny.npy"
predictions_file_path_1 = "loss-val-loss-predictions-z0-z10/Predictions-z0-Johnny.txt"

loss_1 = np.load(loss_file_path_1)
val_loss_1 = np.load(val_loss_file_path_1)
predictions_1 = np.loadtxt(predictions_file_path_1, skiprows=1)
true_A_s_1 = predictions_1[:, 0]
predicted_A_s_1 = predictions_1[:, 1]
A_s_error_1 = predictions_1[:, 2]

loss_file_path_2 = "loss-val-loss-predictions-z0-z10/loss-z10-Johnny.npy"
val_loss_file_path_2 = "loss-val-loss-predictions-z0-z10/val-loss-z10-Johnny.npy"
predictions_file_path_2 = "loss-val-loss-predictions-z0-z10/Predictions-z10-Johnny.txt"

loss_2 = np.load(loss_file_path_2)
val_loss_2 = np.load(val_loss_file_path_2)
predictions_2 = np.loadtxt(predictions_file_path_2, skiprows=1)
true_A_s_2 = predictions_2[:, 0]
predicted_A_s_2 = predictions_2[:, 1]
A_s_error_2 = predictions_2[:, 2]

'''scaling A_s with 1e9'''
true_A_s_scaled_1 = true_A_s_1 * 1e9
predicted_A_s_scaled_1 = predicted_A_s_1 * 1e9
A_s_error_scaled_1 = A_s_error_1 * 1e9

true_A_s_scaled_2 = true_A_s_2 * 1e9
predicted_A_s_scaled_2 = predicted_A_s_2 * 1e9
A_s_error_scaled_2 = A_s_error_2 * 1e9

x_1 = np.arange(1, len(loss_1) + 1)
x_2 = np.arange(1, len(loss_2) + 1)

'''obtaining four plots'''
fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

axes[0, 0].scatter(x_1, loss_1, label="Loss", color="blue", alpha=0.6)
axes[0, 0].scatter(x_1, val_loss_1, label="Validation Loss", color="orange", alpha=0.6)
axes[0, 0].set_title("Loss and Validation Loss Curve ($z=0$)", fontsize=20)
axes[0, 0].set_xlabel("Epoch", fontsize=18)
axes[0, 0].set_ylabel("Loss", fontsize=18)
axes[0, 0].grid(True, linestyle="--", alpha=0.6)
axes[0, 0].legend(fontsize=15)
axes[0, 0].tick_params(axis='both', which='major', labelsize=16)
axes[0, 0].set_ylim(0, 0.08)  

'''errorbars for predictions of z=0'''
axes[0, 1].errorbar(
    true_A_s_scaled_1, 
    predicted_A_s_scaled_1, 
    yerr=A_s_error_scaled_1, 
    fmt='o', 
    color="green", 
    ecolor="black", 
    elinewidth=1.5, 
    capsize=4, 
    alpha=0.6, 
    label="$A_s^{predicted}$ vs $A_s^{true}$"
)
axes[0, 1].plot(true_A_s_scaled_1, true_A_s_scaled_1, color="red", linestyle="--", label="Ideal $A_s^{true}$ = $A_s^{predicted}$")
axes[0, 1].set_title("$A_s^{true}$ vs $A_s^{predicted}$ ($z=0$)", fontsize=20)
axes[0, 1].set_xlabel("$A_s^{true} \\, (\\times 10^9)$", fontsize=18)
axes[0, 1].set_ylabel("$A_s^{predicted} \\, (\\times 10^9)$", fontsize=18)
axes[0, 1].grid(True, linestyle="--", alpha=0.6)
axes[0, 1].legend(fontsize=15, loc="upper left")
axes[0, 1].tick_params(axis='both', which='major', labelsize=16)

axes[1, 0].scatter(x_2, loss_2, label="Loss", color="blue", alpha=0.6)
axes[1, 0].scatter(x_2, val_loss_2, label="Validation Loss", color="orange", alpha=0.6)
axes[1, 0].set_title("Loss and Validation Loss Curve ($z=10$)", fontsize=20)
axes[1, 0].set_xlabel("Epoch", fontsize=18)
axes[1, 0].set_ylabel("Loss", fontsize=18)
axes[1, 0].grid(True, linestyle="--", alpha=0.6)
axes[1, 0].legend(fontsize=15)
axes[1, 0].tick_params(axis='both', which='major', labelsize=16)
axes[1, 0].set_ylim(0, 0.08)  

'''errorbars for predictions of z=10'''
axes[1, 1].errorbar(
    true_A_s_scaled_2, 
    predicted_A_s_scaled_2, 
    yerr=A_s_error_scaled_2, 
    fmt='o', 
    color="green", 
    ecolor="black", 
    elinewidth=1.5, 
    capsize=4, 
    alpha=0.6, 
    label="$A_s^{predicted}$ vs $A_s^{true}$"
)
axes[1, 1].plot(true_A_s_scaled_2, true_A_s_scaled_2, color="red", linestyle="--", label="Ideal $A_s^{true}$ = $A_s^{predicted}$")
axes[1, 1].set_title("$A_s^{true}$ vs $A_s^{predicted}$ ($z=10$)", fontsize=20)
axes[1, 1].set_xlabel("$A_s^{true} \\, (\\times 10^9)$", fontsize=18)
axes[1, 1].set_ylabel("$A_s^{predicted} \\, (\\times 10^9)$", fontsize=18)
axes[1, 1].grid(True, linestyle="--", alpha=0.6)
axes[1, 1].legend(fontsize=15, loc="upper left")
axes[1, 1].tick_params(axis='both', which='major', labelsize=16)

plt.show()

#%%