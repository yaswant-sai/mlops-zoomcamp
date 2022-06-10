import os
import datetime
import pickle
import requests
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

def get_year_month(year, month):
    train_month = month - 2
    val_month = month - 1
    train_year = year
    val_year = year
    if (train_month) < 1:
        train_month = 12 - train_month
        train_year -= 1
    if (val_month) < 1:
        val_month = 12 - val_month
        val_year -= 1
    return train_year, train_month, val_year, val_month
    
@task
def get_paths(date):
    if date is None:
        date = datetime.date.today()
        train_year,  train_month, val_year, val_month= get_year_month(date.year, date.month)
    else:
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
        train_year,  train_month, val_year, val_month= get_year_month(date.year, date.month)
    train_path = f"./data/fhv_tripdata_{train_year}-{train_month:02d}.parquet"
    val_path = f"./data/fhv_tripdata_{val_year}-{val_month:02d}.parquet"
    return train_path, val_path

@task
def read_data(path):
    if not os.path.exists(path):
        file_name = os.path.split(path)[-1]
        url = f'https://nyc-tlc.s3.amazonaws.com/trip+data/{file_name}'
        r = requests.get(url, allow_redirects=True)
        open(f'./data/{file_name}', 'wb').write(r.content)
    df = pd.read_parquet(path)#.head(10000)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
def main(date=None):

    categorical = ['PUlocationID', 'DOlocationID']

    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)

    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)

    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    with open(f".\models\model-{date}.pkl", "wb") as f_out:
        pickle.dump(lr, f_out)

    with open(f".\models\dv-{date}.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)

main(date="2021-08-15")