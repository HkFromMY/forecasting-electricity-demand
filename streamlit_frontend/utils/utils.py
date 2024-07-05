import streamlit as st
import tensorflow as tf
import pandas as pd
import math
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

def create_fourier(date_index):
    fourier = CalendarFourier(
        freq='D',
        order=1
    )
    dp = DeterministicProcess(
        index=date_index,
        constant=True,
        order=1,
        seasonal=True,
        additional_terms=[fourier],
        drop=True
    )

    return dp

@st.cache_data
def encoding_data(df):
    # label encoding
    precip_encoder = OneHotEncoder(sparse_output=False)
    holiday_encoder = OneHotEncoder(sparse_output=False)
    summary_encoder = OneHotEncoder(sparse_output=False)
    household_encoder = LabelEncoder()
    stdor_encoder = OneHotEncoder(sparse_output=False)
    acorn_encoder = LabelEncoder()

    precip_labels = precip_encoder.fit_transform(df[['precipType']])
    holiday_labels = holiday_encoder.fit_transform(df[['Type']])
    summary_labels = summary_encoder.fit_transform(df[['summary']])
    household_labels = household_encoder.fit_transform(df[['LCLid']])
    stdor_labels = stdor_encoder.fit_transform(df[['stdorToU']])
    acorn_labels = acorn_encoder.fit_transform(df[['Acorn']])

    precip_labels = pd.DataFrame(precip_labels, columns=precip_encoder.get_feature_names_out()).astype('int8')
    holiday_labels = pd.DataFrame(holiday_labels, columns=holiday_encoder.get_feature_names_out()).astype('int8')
    summary_labels = pd.DataFrame(summary_labels, columns=summary_encoder.get_feature_names_out()).astype('int8')
    household_labels = pd.DataFrame(household_labels, columns=['household_label']).astype('int8')
    stdor_labels = pd.DataFrame(stdor_labels, columns=stdor_encoder.get_feature_names_out()).astype('int8')
    acorn_labels = pd.DataFrame(acorn_labels, columns=['acorn_label']).astype('int8')

    encoded_df = pd.concat([df, precip_labels, holiday_labels, summary_labels, household_labels, stdor_labels, acorn_labels], axis=1)
    encoded_df = encoded_df.drop(['precipType', 'Type', 'summary', 'LCLid', 'stdorToU', 'Acorn'], axis=1)

    return encoded_df, household_encoder

@st.cache_data
def generate_fourier(df):
    dps = {}

    all_household_ids = df['household_label'].unique()
    households_grp = df.groupby('household_label')
    training_df = []
    testing_df = []

    for household_id in all_household_ids:
        # get the household from the entire df
        print("Household id:", household_id)
        singlehousehold_df = households_grp.get_group(household_id).sort_values(by='tstp', ascending=True)

        # split into train, test
        cutoff_point = math.floor(singlehousehold_df.shape[0] * 0.9)
        training_singlehousehold_df = singlehousehold_df.iloc[:cutoff_point].reset_index()
        testing_singlehousehold_df = singlehousehold_df.iloc[cutoff_point:].reset_index()

        # create fourier features
        fourier_obj = create_fourier(training_singlehousehold_df.set_index('tstp').asfreq('h').index)
        fourier_train_features = fourier_obj.in_sample().reset_index().drop(['tstp'], axis=1)
        training_singlehousehold_df = pd.concat([training_singlehousehold_df, fourier_train_features], axis=1)
        fourier_test_features = fourier_obj.out_of_sample(testing_singlehousehold_df.shape[0]).reset_index().drop(['index'], axis=1)
        testing_singlehousehold_df = pd.concat([testing_singlehousehold_df.reset_index().drop(['index'], axis=1), fourier_test_features], axis=1)
        
        # save the dp object
        dps[household_id] = fourier_obj

        # append to the household_dfs list
        training_df.append(training_singlehousehold_df)
        testing_df.append(testing_singlehousehold_df)

    training_df = pd.concat(training_df, axis=0)
    testing_df = pd.concat(testing_df, axis=0)

    return training_df, testing_df, dps

@st.cache_data
def normalize_train_test(training_df, testing_df):
    # drop unnecessary columns
    training_df = training_df.drop(['index', 'tstp'], axis=1)
    testing_df = testing_df.drop(['level_0', 'tstp'], axis=1)

    # separate into features and target
    X_train = training_df.drop(['energy(kWh/hh)'], axis=1)
    X_test = testing_df.drop(['energy(kWh/hh)'], axis=1)
    y_train = training_df['energy(kWh/hh)']
    y_test = testing_df['energy(kWh/hh)']

    # store the household ids for future evaluation
    test_household_ids = X_test['household_label']
    train_household_ids = X_train['household_label']

    # Scale with MinMax Normalization
    scaler = MinMaxScaler(feature_range=(0, 1)) # scale to 0 and 1
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # reshape for LSTM, CNN-LSTM models
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'test_household_ids': test_household_ids
    }

@st.cache_data
def load_data():
    df = pd.read_csv('data\\24timestep1h.csv', parse_dates=['tstp'])
    df = df.drop(['date'], axis=1) # date column when joined with the holiday

    encoded_df, household_encoder = encoding_data(df)
    training_df, testing_df, dps = generate_fourier(encoded_df)
    cleaned_data = normalize_train_test(training_df, testing_df)

    cleaned_data['test_household_ids'] = household_encoder.inverse_transform(cleaned_data['test_household_ids'])

    return cleaned_data

@st.cache_resource
def load_deterministic_model():
    model = tf.keras.models.load_model('models\\lstm_model.h5')

    return model

def pinball_loss(y, y_hat, alpha):
    """
    Loss function for the probabilistic LSTM
    """

    error = (y - y_hat)
    loss = tf.keras.backend.mean(
        tf.keras.backend.maximum(alpha * error, (alpha - 1) * error), 
        axis=-1
    )

    return loss

@st.cache_resource 
def load_probabilistic_model():
    model = tf.keras.models.load_model(
        'models\\quantile_lstm.h5',
        compile=False,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "output_20": lambda y, y_hat: pinball_loss(y, y_hat, 0.2),
            "output_50": lambda y, y_hat: pinball_loss(y, y_hat, 0.5),
            "output_80": lambda y, y_hat: pinball_loss(y, y_hat, 0.8)
        }
    )
    
    return model