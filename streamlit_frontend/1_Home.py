import streamlit as st
from streamlit_dynamic_filters import DynamicFilters
import pandas as pd 

@st.cache_data
def load_data():
    """
        Load big data from the CSV files and cache it for better performance
    """
    data = pd.read_csv('data\\24timestep1h.csv', parse_dates=['tstp'])

    return data

def home_page():

    # filter by household id and date for energy consumption data line chart
    df = load_data()
    dynamic_filters = DynamicFilters(df, filters=['LCLid', 'year', 'month'])
    filtered_df = dynamic_filters.filter_df()
    df_size_mb = filtered_df.memory_usage(deep=True).sum() / (1024 ** 2)

    st.header("Visualizing the energy consumption")
    dynamic_filters.display_filters()
    if df_size_mb > 500:
        st.write("Please select proper filters to visualize the data")
    else:
        st.line_chart(
            filtered_df[['tstp', 'energy(kWh/hh)']], 
            x='tstp', 
            y='energy(kWh/hh)',
            x_label='Datetime',
            y_label='Energy (kWh/hh)'
        )

    st.header("View data in tabular form")
    if df_size_mb > 500:
        st.write("Please select proper filters to view the data")
    else:
        st.dataframe(filtered_df, hide_index=True)

home_page()