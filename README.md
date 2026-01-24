# CDE-Multivariate-TMSS-data-MDN-

## Title: Novel developments of Mixture Density Networks: A supervised approach for learning multivariate time series

## Abstract: 

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
- Data generation: `experiments/data_generation.py`
- Model training: `experiments/train_mdn.py`
- Evaluation: `evaluation/compute_wd.py`


