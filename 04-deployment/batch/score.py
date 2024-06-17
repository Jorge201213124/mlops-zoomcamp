#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pickle

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


def read_data(filename, categorical, year, month):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    return df


def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def apply_model(input_file, output_file, year, month):

    categorical = ['PULocationID', 'DOLocationID']
    df = read_data(input_file, categorical, year, month)
    
    dv, model = load_model()
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print("STD", y_pred.std())
    print("MEAN", y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['prediction'] = y_pred
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    taxi_type = 'yellow'


    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{year:04d}-{month:02d}.parquet'


    apply_model(
        input_file=input_file, 
        output_file=output_file,
        year=year,
        month=month
    )


if __name__ == "__main__":
    run()
