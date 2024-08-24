# Optimizing Residential Electrical Consumption with Machine Learning and Deep Learning 

## Problem Statement
1. Global net energy consumption is increasing yearly which also increases the prices of the fuel sources globally, ultimately leading to energy crisis problem if neglected.
2. Some countries are transitioning towards renewable energy sources like solar and wind power, but the unpredictability of the energy generation remains an obstacle.
3. To motivate the consumer to reduce energy consumption during peak periods, some countries are implementing demand response programme such as Time of Use (ToU) that adjusts electricity pricing according to the load demand. However, the structure of the demand response programme is rigid due to the challenges faced in tailoring the incentive mechanisms to the unique requirements of each consumer group. This makes the scheduling of the pricing complex because the utility providers need to consider many factors.

## Project Aim
To optimize the residential energy consumption with forecasted energy demand through multivariate time-series forecasting using machine learning and deep learning.

## Project Objectives
1. To conduct comprehensive literature review on the existing approaches and models implemented to accurately forecast the energy demand.
2. To develop suitable predictive models that can accurately forecast the electricity consumption in short-term basis in 1-hour resolution for the next 72 hours.
3. To analyse the insights from the forecasted values to optimize energy consumption in terms of resource allocation and electrical energy management on operational level.  

## Target Users
1. Government Policy Maker - Help them in infrastructure planning after understanding the load demand for that geographic area 
2. Private Energy Supplier - Help in better formulating the electricity pricing a day in advance to minimize operational cost.

## Project Deliverables
1. Data collection
2. Exploratory Data Analysis (EDA)
3. Data Pre-Processing
4. Model Development
5. Model Evaluation

## Technology Used
1. NumPy
2. Pandas
3. Statsmodels
4. Matplotlib
5. Seaborn
6. Keras
7. Tensorflow
8. Scikit-learn
9. Streamlit

## Metadata
- Dataset is obtained from ![Kaggle](https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london) which is published by London Data Stores.
- It contains half-hourly energy consumption measures collected from 5567 households in London between November 2011 and February 2014.
- It also includes additional details like weather dataset, holidays, and demographic information according ot A Classification of Residential Neighbourhoods (ACORN).

## Data Pre-processing & Feature Engineering 
![image](https://github.com/user-attachments/assets/aeaae311-87e5-4275-981e-49e1d5c54b85)  
The image above is the overall workflow for the data pre-processing and feature engineering phase. The details of the processes can be referred in the notebooks:
1. `Data Preprocessing.ipynb`
2. `Feature Engineering.ipynb`

## Model Development 
The model developed are from 2 distinct approaches, namely deterministic and probabilistic approach. Deterministic approach involves forecasting single exact value of future occurrences while the probabilistic approach predicts the probability distribution of the future instances. In this project, the Quantile Regression method is used for the probabilistic approach method. The model development processes can be referred in `Multiple Household` section in both `Deterministic Forecasting.ipynb` and `Probabilistic Forecasting.ipynb` notebooks. For the Quantile Regression, the selected quantiles to be forecasted is 10-th, 30-th, 50-th, 70-th, and 90-th quantiles respectively. Thus, the main difference between the deterministic and probabilistic approaches is that the probabilistic models has 5 outputs instead of 1 which corresponds to each of the quantile.  

## Hyperparameter Tuning 
The details of the processes can be found in the `Deterministic Forecasting.ipynb` and `Probabilistic Forecasting.ipynb` notebooks respectively.

### Machine Learning Models
- `XGBoost`, `Quantile-XGBoost`, `Quantile-LightGBM` had implemented GridSearch to search for the best hyperparameter set. 3-Fold cross validation is used to ensure that the results from the search is reliable.
- `Random Forest` had implemented RandomSearch as the training hour for single model takes around 30 minutes which can be impractical to implement GridSearch to search for all possible hyperparameters. 

### Deep Learning Models
- All deep learning models had implemented Hyperband strategy to search for more combination of hyperparameters while maintaining efficient search time. 

## Model Evaluation
### Deterministic Model
From the comparison in `Model Evaluation.ipynb` notebook, the best model is `CNN-LSTM` as it has the lowest error metrics like RMSE, MSE, MAE, and MAPE. Looking at the residual distribution, the model is not biased towards any instance as the residual distribution follows the normal shape distribution which is reflected in the bell-curved shape. 

### Probabilistic Model 
In the same notebook `Model Evaluation.ipynb`, the best model for probabilistic approach is `Quantile-XGBoost` which is reflected in total pinball losses which indicates that it cna estimate the overall quantiles better. The residual distribution of the model is slightly skewed to the left, which indicates the model tend to underestimate some of the outlier instances in terms of the forecasted median (50-th quantile). 

## Model Deployment 
The model is deployed to Streamlit application, which can be started using the following commands:  
1. Run `git clone https://github.com/HkFromMY/energy-forecast-streamlit.git` (stored in another repository due to large files)
2. Follows the instruction on `README.md` on that repository to run the application.
