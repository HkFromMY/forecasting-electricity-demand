import streamlit as st
st.set_page_config(page_title="Deterministic Model", page_icon="📈", layout="wide")

from utils.utils import (
    load_data,
    load_deterministic_model
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_forecast_by_household(households, household_id, y_pred, y_test, number_of_slices=3, steps=168, skip=0):
    # plot the forecast plot by household
    # arrange the household ids
    households = np.expand_dims(households, axis=1)
    y_pred = np.expand_dims(y_pred.ravel(), axis=1)
    y_test = np.expand_dims(y_test.to_numpy(), axis=1)
    integrated = np.hstack((households, y_pred, y_test))
    integrated = pd.DataFrame(integrated, columns=['household_ids', 'y_pred', 'y_test'])

    # filter the data by household
    filtered_df = integrated.loc[integrated['household_ids'] == household_id]

    # plot the chart
    fig, axes = plt.subplots(number_of_slices, 1, sharey=True, figsize=(14, 10))
    fig.suptitle(f'Plot for {household_id}')
    timesteps = np.arange(0, steps)

    for i in range(number_of_slices):
        start_idx = 0 + (steps * i) + (skip)
        end_idx = steps * (1 + i) + (skip)
        sampled_test = filtered_df.iloc[start_idx:end_idx]['y_test']
        sampled_pred = filtered_df.iloc[start_idx:end_idx]['y_pred']
        mape_score = mean_absolute_percentage_error(sampled_test, sampled_pred)
        
        sns.lineplot(x=timesteps, y=sampled_test, label="truth-value", marker='o', alpha=0.3, ax=axes[i])
        sns.lineplot(x=timesteps, y=sampled_pred, label="pred-value", marker='o', ax=axes[i])
        axes[i].set_title(f'MAPE: {round(mape_score, 2)}')
        axes[i].set_ylabel('Energy (kWh/hh)')

    plt.tight_layout() 

    return fig

@st.cache_data
def predict(X_test):
    model = load_deterministic_model()

    y_pred = model.predict(X_test)

    return y_pred, y_pred.ravel()

def deterministic_model_page():
    df = load_data()

    # Testing data
    X_test = df['X_test']
    y_test = df['y_test']

    # Predictions
    y_pred, y_pred_flatten = predict(X_test)
    test_household_ids = df['test_household_ids']

    # calculate nad print the evaluation metrics
    mse = mean_squared_error(y_pred_flatten, y_test)
    rmse = mean_squared_error(y_pred_flatten, y_test, squared=False)
    mae = mean_absolute_error(y_pred_flatten, y_test)
    r2 = r2_score(y_pred_flatten, y_test)
    mape = mean_absolute_percentage_error(y_pred_flatten, y_test)

    st.header("Deterministic Model (LSTM)")
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

deterministic_model_page()