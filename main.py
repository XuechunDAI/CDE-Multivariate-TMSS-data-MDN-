## Main 

"""
This script reproduces the results shown in Section Results.
"""

from google.colab import drive
drive.mount('/content/drive')  # Mount Google Drive

# The path in Drive
load_data_ptimes(p=6, n=10000, filename='/content/drive/MyDrive/thesis/pre-code/A.npy')

import numpy as np
#from model_setup_2D import RNN, train
import torch
import matplotlib.pyplot as plt
#from load_data import data_process, load_data, load_data_ptimes
#from data_view import contourf_shower, loss_curve_shower, plot_WD2
#from contourRd import transport, plot_results
#from baseline_methods import compute_psi_matrices



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

np.random.seed(42)
# Generate samples from GMM
samples = gmm.sample(1000)

#step 2: transport and show results
results = transport(samples, nR=40, nS=25, d=2)
plot_results(samples, results)



# Figure 1b

# Generate data
Xt = load_data_p_sn(p = 6, n = 10000, filename='A.npy')
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

np.random.seed(42)
# Generate samples from GMM
samples = gmm.sample(1000)

#step 2: transport and show results
results = transport(samples, nR=40, nS=25, d=2)
plot_results_appr(samples, results, q1_points_true, q2_points_true, q3_points_true)


# Figure 1c

# Generate data
Xt = load_data_p_t(p = 6, n = 10000, filename='A.npy')
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
err_sample = multivariate_t.rvs(loc=np.zeros(2), shape=cov, df=df, size=sample_size)

for i in range(sample_size):
    test_X_sample[i] = true_test_X + err_sample[i]

Qj3D1 = test_X_sample  # for convenience
results_true = transport(Qj3D1, nR=40, nS=25, d=2)
q1_points_true, q2_points_true, q3_points_true = results_true['q1_points'], results_true['q2_points'], results_true['q3_points']

np.random.seed(42)
# Generate samples from GMM (2000 samples)
samples = gmm.sample(2000)

#step 2: transport and show results
results = transport(samples, nR=40, nS=50, d=2)
plot_results_appr(samples, results)


# Figure 1d

# Generate data
Xt = load_gaus_mix_2p(p = 6, n = 10000, filename='A.npy')
X_train, y_train, X_val, y_val = data_process(Xt, p = 6, r = 0.2)
print(Xt.shape)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

test_X = X_val[-5]
test_y = y_val[-5]
print(test_X.shape)
print(test_X)

set_seed(42)
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
    true_test_X += A[i] @ test_X[-(i+1)]

np.random.seed(42) ###

test_X_sample = np.zeros((1000, 2))
for i in range(1000):
    if np.random.rand() < pi:
        test_X_sample[i] = true_test_X + np.random.multivariate_normal(mu1, cov)
    else:
        test_X_sample[i] = true_test_X + np.random.multivariate_normal(mu2, cov)

Qj3D1 = test_X_sample  # for convenience
results_true = transport(Qj3D1, nR=40, nS=25, d=2)
q1_points_true, q2_points_true, q3_points_true = results_true['q1_points'], results_true['q2_points'], results_true['q3_points']

np.random.seed(42)
# Generate samples from GMM (2000 samples)
samples = gmm.sample(2000)

#step 2: transport and show results
results = transport(samples, nR=40, nS=50, d=2)
plot_results_appr(samples, results)



# Dynamics
# Figure 2a, 3b

# Generate data
data_dict = load_arma(p = 1, n = 10000)
Xt = data_dict["Xt"]
eps = data_dict["eps"]

X_train, y_train, X_val, y_val = data_process(Xt, p = 10, r = 0.2)

print(Xt.shape, eps.shape)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

test_X = X_val[-5]
test_y = y_val[-5]
print(test_X.shape)

set_seed(42)
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

mean_true_testX = Phi @ test_X[-1] + Theta @ eps[-6] # compute the true mean

np.random.seed(42)
# Generate samples from GMM
samples = gmm.sample(2000)

#step 2: transport and show results
results = transport(samples, nR=40, nS=50, d=2)
plot_results_admean(samples, results)

## Figure 3b

len_train = X_train.shape[0] - 1 + X_train.shape[1]
data_train = Xt[:len_train, :]
df = pd.DataFrame({
    'dim1': data_train[:, 0],
    'dim2': data_train[:, 1]
})

model_varma = VARMAX(df, order=(1, 1))
results_varma = model_varma.fit(disp=True)
mu_param = results_varma.params[:2].values.reshape(2, )
ar_coefs = results_varma.params[2:6].values.reshape(2, 2)
ma_coefs = results_varma.params[6:10].values.reshape(2, 2)
L = np.array([[results_varma.params['sqrt.var.dim1'], 0],
              [results_varma.params['sqrt.cov.dim1.dim2'], results_varma.params['sqrt.var.dim2']]])
Sigma_param = L @ L.T

# KDE
standardized_residuals = calculate_residuals_varma(X_train, Psi_list, Sigma_param)
kde_residuals = KDEMultivariate(standardized_residuals,
                               bw='normal_reference',
                               var_type='cc')

# prepare data
X_val_200 = X_val[0:200, :, :]
distances_200 = compute_wasserstein_varma(X_val_200, Phi, Theta, Sigma, mu_param, ar_coefs, ma_coefs, Sigma_param, m=1000)

ranks = [('rank 10', 'q1'), ('rank 21', 'q2'), ('rank 33', 'q3')]
for rank_name, rank_key in ranks:
    plot_WD2(distances_200, rank_name, rank_key)

mdn_data = [distances_200['mdn'][key] for key in ['q1', 'q2', 'q3']]
plt.figure()
plt.boxplot(mdn_data)
plt.xticks([1, 2, 3], ['rank10', 'rank21', 'rank33'])
plt.title('Distribution of W2 distances at different ranks')
plt.ylabel('Wasserstein Distance')
plt.show()


# Figure 2b, 3c

# Generate data
data_dict = load_vol_VAR(p = 1, n = 10000)
Xt = data_dict["Xt"]
a = data_dict["a"]
Sigma = data_dict["Sigma"]

X_train, y_train, X_val, y_val = data_process(Xt, p = 1, r = 0.2)
print(Xt.shape, a.shape, Sigma.shape)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

test_X = X_val[-5]
test_y = y_val[-5]
print(test_X.shape)

set_seed(42)
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

mean_true_testX = Phi @ Xt[9994].flatten()
Sigma_true_testX = (C @ C.T + 
                    A @ np.outer(a[9994], a[9994]) @ A.T + 
                    B @ Sigma[9994] @ B.T)

np.random.seed(42)
# Generate samples from GMM
samples = gmm.sample(2000)

#step 2: transport and show results
results = transport(samples, nR=40, nS=50, d=2)
plot_results(samples, results)

## 3c

data = np.squeeze(X_train)
data_Sigma = Sigma[0:len(X_train)]
pd.DataFrame(data).to_csv("data.csv", index=False)

%%R
data_r <- read.csv("data.csv")
data_r <- as.matrix(data_r)

spec <- bekk_spec()
model_bekk <- bekk_fit(spec, data_r, QML_t_ratios = TRUE, max_iter = 50)
C_matrix <- model_bekk$C0
A_matrix <- model_bekk$A
G_matrix <- model_bekk$G

T <- nrow(model_bekk$H_t)
list_Sigma_param <- array(NA, dim = c(2, 2, T))
for (t in 1:T) {
  list_Sigma_param[, , t] <- matrix(model_bekk$H_t[t, ], nrow = 2, byrow = TRUE)
}
# ===== finish R environment =====

# transform to Python object
from rpy2.robjects import r

r('''
params <- model_bekk$est.params
C_param <- model_bekk$C0
A_param <- model_bekk$A
G_param <- model_bekk$G
''')

C_param = np.squeeze(np.array(r['C_param']))
A_param = np.squeeze(np.array(r['A_param']))
G_param = np.squeeze(np.array(r['G_param']))

r('''
list_Sigma_param <- list_Sigma_param
''')

list_Sigma_param = np.array(r['list_Sigma_param'])

# unconditional covariance
Sigma_uncond = list_Sigma_param[:, :, -1].copy()

for iter in range(1000):
    Sigma_new = C_param.T @ C_param + A_param @ Sigma_uncond @ A_param.T + G_param @ Sigma_uncond @ G_param.T
    if np.max(np.abs(Sigma_new - Sigma_uncond)) < 1e-8:
        break

    Sigma_uncond = Sigma_new.copy()

# KDE
standardized_residuals = calculate_residuals_bekk(data, list_Sigma_param)
kde_residuals = KDEMultivariate(standardized_residuals,
                               bw='normal_reference',
                               var_type='cc')

# prepare data
X_val_200 = np.squeeze(X_val[0:200, :, :])
distances_200 = compute_wasserstein_bekk(X_val_200, m=1000)

ranks = [('rank 10', 'q1'), ('rank 21', 'q2'), ('rank 33', 'q3')]
for rank_name, rank_key in ranks:
    plot_WD2(distances_200, rank_name, rank_key)

mdn_data = [distances_200['mdn'][key] for key in ['q1', 'q2', 'q3']]
plt.figure()
plt.boxplot(mdn_data)
plt.xticks([1, 2, 3], ['rank10', 'rank21', 'rank33'])
plt.title('Distribution of W2 distances at different ranks')
plt.ylabel('Wasserstein Distance')
plt.show()


# Figure 2c, 3d

# Generate data
data_dict = load_data_nonlinear(n = 10000)
Xt = data_dict["Xt"]
err = data_dict["err"]

X_train, y_train, X_val, y_val = data_process(Xt, p = 1, r = 0.2)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

test_X = X_val[-5]
test_y = y_val[-5]

set_seed(42)
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
def con_mean(test):

    X1, X2 = np.squeeze(test)
    normX = np.sqrt(X1**2 + X2**2)
    f1 = 0.5 * np.sin(X1 + X2) + 0.25 * np.tanh(normX)
    f2 = (1/3) * np.cos(X1 - X2) + 0.2 * X1 / (1 + normX)

    con_mean = np.array([f1, f2])

    return con_mean

def con_cov(test):

    X1, X2 = np.squeeze(test)
    normX = np.sqrt(X1**2 + X2**2)
    noise_scale = 0.4 * (np.sin(normX / 5) / (2 + normX**2) + 0.5)

    cov = np.eye(2)
    con_cov = (noise_scale ** 2) * cov

    return con_cov

true_test_X = con_mean(test_X) # true mean

test_X_sample = np.zeros((1000, 2))
err_sample = np.zeros((1000, 2))
X1, X2 = test_X[0][0], test_X[0][1]
normX = np.sqrt(X1**2 + X2**2)
noise_scale = 0.4 * (np.sin(normX / 5) / (2 + normX**2) + 0.5)

for h in range(1000):
  u = np.random.rand()
  if u < pi1:
    err_sample[h] = np.random.multivariate_normal(mu1, cov)
  elif u < pi1 + pi2:
    err_sample[h] = np.random.multivariate_normal(mu2, cov)
  else:
    err_sample[h] = np.random.multivariate_normal(mu3, cov)

  test_X_sample[h] = true_test_X + noise_scale * err_sample[h]

Qj3D1 = test_X_sample  # for convenience
results_true = transport(Qj3D1, nR=40, nS=25, d=2)

q1_points_true, q2_points_true, q3_points_true = results_true['q1_points'], results_true['q2_points'], results_true['q3_points']

np.random.seed(42)
# Generate samples from GMM
samples = gmm.sample(1000)

#step 2: transport and show results
results = transport(samples, nR=40, nS=25, d=2)
plot_results_appr(samples, results)

## Figure 3d

len_train = X_train.shape[0] - 1 + X_train.shape[1]
print(len_train)
data_train = Xt[:len_train, :]
df = pd.DataFrame(data_train)

model_param = VAR(df)
results_param = model_param.fit(1)

A_param = np.squeeze(results_param.coefs)
Sigma_param = results_param.sigma_u       # Parametric results

standardized_residuals = calculate_residuals(data_train, A_param, Sigma_param)

kde_residuals = KDEMultivariate(standardized_residuals,
                               bw='normal_reference',
                               var_type='cc')                  # apply KDE

# prepare data
X_val_200 = np.squeeze(X_val[0:200, :, :])
distances_200 = compute_wasserstein_nonl(X_val_200, A_param, Sigma_param, m=1000)

ranks = [('rank 10', 'q1'), ('rank 21', 'q2'), ('rank 33', 'q3')]
for rank_name, rank_key in ranks:
    plot_WD2(distances_200, rank_name, rank_key)

mdn_data = [distances_200['mdn'][key] for key in ['q1', 'q2', 'q3']]
plt.figure()
plt.boxplot(mdn_data)
plt.xticks([1, 2, 3], ['rank10', 'rank21', 'rank33'])
plt.title('Distribution of W2 distances at different ranks')
plt.ylabel('Wasserstein Distance')
plt.show()



# Figure 3a

# Generate data
Xt = load_data(n = 10000)
X_train, y_train, X_val, y_val = data_process(Xt, p = 1, r = 0.2)
print(Xt.shape)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

test_X = X_val[-5]
test_y = y_val[-5]

set_seed(42)
# Define model
torch.set_default_dtype(torch.float64)
model = RNN(gause_mixture_n=10)

# Train model
model, tl, vl = train(model, X_train, y_train, X_val, y_val, lr=3e-4, batch_size=128, epochs=10)
loss_curve_shower(tl, vl)

# parametric estimation
data_train = X_train.reshape(X_train.shape[0], X_train.shape[2])
df = pd.DataFrame(data_train)
model_param = VAR(df)
results_param = model_param.fit(1)

A_param = np.squeeze(results_param.coefs)
Sigma_param = results_param.sigma_u

# KDE
standardized_residuals = calculate_residuals(data_train, A_param, Sigma_param)
kde_residuals = KDEMultivariate(standardized_residuals,
                               bw='normal_reference',
                               var_type='cc')

# prepare data
A = np.array([[0.2, -0.6], [0.3, 1.1]])
Sigma = np.array([[1.0, 0.3], [0.3, 1]])
X_val_200 = np.squeeze(X_val[0:200, :, :])
distances_200 = compute_wasserstein(X_val_200, A, Sigma, A_param, Sigma_param, m=1000)

ranks = [('rank 10', 'q1'), ('rank 21', 'q2'), ('rank 33', 'q3')]
for rank_name, rank_key in ranks:
    plot_WD2(distances_200, rank_name, rank_key)

mdn_data = [distances_200['mdn'][key] for key in ['q1', 'q2', 'q3']]
plt.figure()
plt.boxplot(mdn_data)
plt.xticks([1, 2, 3], ['rank10', 'rank21', 'rank33'])
plt.title('Distribution of W2 distances at different ranks')
plt.ylabel('Wasserstein Distance')
plt.show()


# Figure 4a













