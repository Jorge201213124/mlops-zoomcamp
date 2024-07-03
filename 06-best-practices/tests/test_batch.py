# Libraries
import pandas as pd
import batch_refactored
from datetime import datetime


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


def test_prepare_data():
    # Prepare inputs
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),     
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    categorical = ['PULocationID', 'DOLocationID']

    # Call function
    prepared_data = batch_refactored.prepare_data(df, categorical)

    assert prepared_data.shape == (3,5)
