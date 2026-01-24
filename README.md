# CDE-Multivariate-TMSS-data-MDN-

## Title: Novel developments of Mixture Density Networks: A supervised approach for learning multivariate time series

## Methodology: 

We use a Mixture Density Network (MDN), with a recurrent neural network to perform conditional density estimation for multivariate time series. Specifically, it maps past observations to the parameters of a Gaussian mixture model as a approximation of the true density. 

We assess the model through 
- `Center-outward quantile contour plots` in two dimensional cases; 
- `Wasserstein distance` as a quantitative metric in higher dimensions. Evaluation are conducted between the proposed model and two baseline estimation method parametric estimation and Kernel density estimation. 

## How to Run: 

### Environment
- PyTorch >= 1.13

### Install Dependencies
```bash
pip install -r requirements.txt

```markdown
- Data generation: `load_data.py`
- Model: `model_setup_2D.py` and `model_setup_higher.py` 
- Parametric Models & KDE: `baseline_methods.py`
- Visualization: `data_view.py` 


All datasets used in the thesis are simulated. To generate the datasets, run:
```bash
python load_data.py

To reproduce the main results of the thesis with default settings, run:
```bash
python main.py





