## Main 

"""
This script reproduces the results shown in Section Results.
"""

from google.colab import drive
drive.mount('/content/drive')  # Mount Google Drive

# The path in Drive
load_data_ptimes(p=6, n=10000, filename='/content/drive/MyDrive/thesis/pre-code/A.npy')

import numpy as np
#from model_setup import RNN, train
import torch
import matplotlib.pyplot as plt
#from data_loader import data_process, load_data, load_data_ptimes
#from data_view import contourf_shower, loss_curve_shower
#from contourRd import transport, plot_results



# Figure 1a 

#step 1: generate data
Xt = load_data_ptimes(p = 6, n = 10000, filename='A.npy')
X_train, y_train, X_val, y_val = data_process(Xt, p = 6, r = 0.2)
print(Xt.shape)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

test_X = X_val[-5]
test_y = y_val[-5]

# Define model
torch.set_default_dtype(torch.float64)
model = RNN(gause_mixture_n=10)

# Train model
model, tl, vl = train(model, X_train, y_train, X_val, y_val, lr=3e-4, batch_size=128, epochs=10)
loss_curve_shower(tl, vl)

# get Gaussian mixture model from test_x
gmm = model.predict_model(test_X)

print(gmm)

# Plot contourf of GMM
contourf_shower(gmm)

# Generate samples from GMM
samples = gmm.sample(1000)

#step 2: transport and show results
results = transport(samples, nR=40, nS=25, d=2)
plot_results(samples, results)



# Figure 1b

# Generate data
Xt = load_data_ptimes(p = 6, n = 10000, filename='A.npy')
X_train, y_train, X_val, y_val = data_process(Xt, p = 6, r = 0.2)
print(Xt.shape)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

test_X = X_val[-5]
test_y = y_val[-5]
print(test_X.shape)

# Define model
torch.set_default_dtype(torch.float64)
model = RNN(gause_mixture_n=10)

# Train model
model, tl, vl = train(model, X_train, y_train, X_val, y_val, lr=3e-4, batch_size=128, epochs=10)
loss_curve_shower(tl, vl)

# get Gaussian mixture model from test_x
gmm = model.predict_model(test_X)

print(gmm)

# Plot contourf of GMM
contourf_shower(gmm)

# Generate sample data for true density estimation
A = np.load('A.npy')
true_test_X = 0
for i in range(6):
    true_test_X += A[i] @ test_X[i]

sample_size = 1000
test_X_sample = np.zeros((sample_size, 2))
err_sample = sn.rmsn(sample_size, xi, Omega, alpha)

for i in range(sample_size):
    test_X_sample[i] = true_test_X + err_sample[i]

Qj3D1 = test_X_sample  # for convenience
results_true = transport(Qj3D1, nR=40, nS=25, d=2)
q1_points_true, q2_points_true, q3_points_true = results_true['q1_points'], results_true['q2_points'], results_true['q3_points']

# Generate samples from GMM
samples = gmm.sample(1000)

#step 2: transport and show results
results = transport(samples, nR=40, nS=25, d=2)
plot_results_appr(samples, results, q1_points_true, q2_points_true, q3_points_true)






