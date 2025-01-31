import numpy as np
from openfermion import get_sparse_operator
from optomat import xx1,xx2,xx3,xxd2
from utils.util_hamil import test0,test,test1,test2,test5

from utils.util_mutualinfo import mutual_information
from utils.util_covar import covariance,covariance_x2,covariance_x21
from ten_network import mps1,cpd1,mps2,cpd2
from utils.util_save import printx,save_parameters
from utils.util_gfro import (obtain_fragment,
                       rotated_hamiltonian,
                       boson_eigenspectrum_sparse,
                       boson_eigenspectrum_full,
                       quad_diagonalization)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# Function to update the plot for each frame
def update(frame, line):
    lam = frame  # Vary `lam` as the frame number
    h_variables = [1 / 2, 1 / 2, lam]

    H = test1(h_variables)  # Generate Hamiltonian
    eigenvalues1, e1 = boson_eigenspectrum_full(H, truncation)

    parameters_list = []  # Temporary storage for this iteration
    for i in range(ranges):
        ee2 = e1[:, i]
        ff1 = ee2.reshape(truncation, truncation)

        # Append properties for current state
        parameters_list.append([
            f'State {i}',
            round(eigenvalues1[i], 7),
            round(mutual_information(ff1, truncation), 7),
            round(covariance(ee2, truncation), 7),
            round(covariance_x2(ee2, truncation), 7),
            round(covariance_x21(ee2, truncation), 7)
        ])

    # Extract energies and mutual information
    energies = np.array(parameters_list)[:, 1].astype(float)
    mutual_info = np.abs(np.array(parameters_list)[:, 2].astype(float))

    # Update plot data
    line.set_data(energies, mutual_info)
    ax.relim()
    ax.autoscale_view()

    return line,


# Plot setup
fig, ax = plt.subplots()
ax.set_title('Energy vs Mutual Information')
ax.set_xlabel('Energy')
ax.set_ylabel('Mutual Information')
line, = ax.plot([], [], 'o', label='Energy vs Mutual Info')

# Parameters for animation
num_frames = 100  # Number of frames, varies `lam` from 0 to 999
frames = np.linspace(0.8, 1.2, num_frames)
truncation = 10
ranges = 99

# Create an animation
ani = FuncAnimation(fig, update, frames=frames, fargs=([line]), blit=False)

# Save the animation to a video file (optional)
ani.save('energy_vs_mutual_info.gif', writer=PillowWriter(fps=15))

# Show the plot interactively
plt.show()
