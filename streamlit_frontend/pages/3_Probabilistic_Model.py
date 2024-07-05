import streamlit as st
st.set_page_config(page_title="Probabilistic Model", page_icon="📈", layout="wide")

from utils.utils import (
    load_data,
    load_probabilistic_model
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    mean_pinball_loss
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import traceback

def plot_forecast_by_household(households, household_id, y_pred, y_test, number_of_slices=3, steps=168, skip=0):
    # plot the forecast plot by household
    # arrange the household ids

    sampled_df = y_pred.copy(deep=True)
    sampled_df['household_ids'] = households 
    sampled_df['y_test'] = y_test.reset_index()['energy(kWh/hh)']

    # filter the data by household
    filtered_df = sampled_df.loc[sampled_df['household_ids'] == household_id]

    # plot the chart
    fig, axes = plt.subplots(number_of_slices, 1, sharey=True, figsize=(14, 10))
    fig.suptitle(f'Plot for {household_id}')
    timesteps = np.arange(0, steps)

    for i in range(number_of_slices):
        start_idx = 0 + (steps * i) + (skip)
        end_idx = steps * (1 + i) + (skip)
        sampled_test = filtered_df.iloc[start_idx:end_idx]['y_test']
        sampled_pred = filtered_df.iloc[start_idx:end_idx][['pred_0.2', 'pred_0.5', 'pred_0.8']]
        mape_score = mean_absolute_percentage_error(sampled_pred['pred_0.5'], sampled_test)
        
        sns.lineplot(x=timesteps, y=sampled_test, label="truth-value", marker='o', alpha=0.3, ax=axes[i])
        sns.lineplot(x=timesteps, y=sampled_pred['pred_0.5'], label="forecasted-median", marker='o', ax=axes[i])
        axes[i].fill_between(x=timesteps, y1=sampled_pred['pred_0.2'], y2=sampled_pred['pred_0.8'], alpha=0.1, color='blue')
        axes[i].set_title(f'MAPE: {round(mape_score, 2)}')
        axes[i].set_ylabel('Energy (kWh/hh)')

    plt.tight_layout() 

    return fig

@st.cache_data
def predict(X_test):
    model = load_probabilistic_model()

    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred).squeeze() # compress to 2-Dimensional array
    y_pred = pd.DataFrame({
        'pred_0.2': y_pred[0].ravel(),
        'pred_0.5': y_pred[1].ravel(),
        'pred_0.8': y_pred[2].ravel()
    })

    return y_pred

def probabilistic_page():
    df = load_data()

    # Testing data
    X_test = df['X_test']
    y_test = df['y_test']

    # Predictions
    y_pred = predict(X_test)
    test_household_ids = df['test_household_ids']

    # calculate nad print the evaluation metrics
    mse = mean_squared_error(y_pred['pred_0.5'], y_test)
    rmse = mean_squared_error(y_pred['pred_0.5'], y_test, squared=False)
    mae = mean_absolute_error(y_pred['pred_0.5'], y_test)
    r2 = r2_score(y_pred['pred_0.5'], y_test)
    mape = mean_absolute_percentage_error(y_pred['pred_0.5'], y_test)

    # pinball losses
    pinball_20 = mean_pinball_loss(y_pred['pred_0.2'], y_test, alpha=0.2)
    pinball_50 = mean_pinball_loss(y_pred['pred_0.5'], y_test, alpha=0.5)
    pinball_80 = mean_pinball_loss(y_pred['pred_0.8'], y_test, alpha=0.8)

    st.header("Probabilistic Model (Quantile-LSTM)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label='MSE', value=round(mse, 4))

    with col2:
        st.metric(label='RMSE', value=round(rmse, 4))

    with col3:
        st.metric(label='MAE', value=round(mae, 4))

    with col4:
        st.metric(label='R2', value=round(r2, 5))
    
    with col5:
        st.metric(label='MAPE', value=round(mape, 4))

    col1, col2, col3 = st.columns(3)
    with col1: 
        st.metric(label='Pinball Loss (20)', value=round(pinball_20, 4))

    with col2:
        st.metric(label='Pinball Loss (50)', value=round(pinball_50, 4))

    with col3:
        st.metric(label='Pinball Loss (80)', value=round(pinball_80, 4))

    st.header("Visualize forecasting by household")
    col1, col2, col3 = st.columns(3)
    with col1:
        households = np.unique(test_household_ids)
        selected_household = st.selectbox("Select Household", households)

    with col2:
        step_size = st.number_input("Step Size", min_value=24, max_value=168, value=168, step=24)

    with col3:
        skip_slice = st.number_input("Skip Slice", min_value=0, value=0, step=3)
    
    try:
        st.pyplot(plot_forecast_by_household(test_household_ids, selected_household, y_pred, y_test, number_of_slices=3, steps=step_size, skip=(skip_slice * step_size)))

    except ValueError:
        st.write("The skipped slices is more than the testing samples! Please adjust the skip slice value.")

probabilistic_page()