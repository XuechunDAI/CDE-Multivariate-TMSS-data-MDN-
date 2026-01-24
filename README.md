# CDE-Multivariate-TMSS-data-MDN-

## Title: Novel developments of Mixture Density Networks: A supervised approach for learning multivariate time series

## Methodology: 

We use a Mixture Density Network (MDN), with a recurrent neural network to perform conditional density estimation for multivariate time series. Specifically, it maps past observations to the parameters of a Gaussian mixture model which serves as an approximation of the true density. 

Model performance is assessed through: 
- Center-outward quantile contour plots in two dimensional cases; 
- Wasserstein distance as a quantitative metric in higher dimensions. The proposed model is compared with two baseline estimation method parametric estimation and Kernel density estimation. 

## How to Run: 

### Environment
- PyTorch >= 1.13

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Code Structure
The main scripts are organized as follows:
- Data generation: `load_data/`
- MDN Model:
  - `model_setup_2D/`
  - `model_setup_higher/` 
- Parametric Models and KDE: `baseline_methods/`
- Computation of center-outward quantiles: `contourRd/`
- Visualization: `data_view/` 

### Data Generation 
All datasets used in the thesis are simulated. To generate the datasets, run:
```bash
python load_data.py
```

### Reproducing the Main Results 
To reproduce the main results of the thesis with default settings, run:
```bash
python main.py
```








