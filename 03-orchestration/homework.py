import pandas as pd
from prefect import flow, task, get_run_logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime
from dateutil.relativedelta import relativedelta
import urllib.request
import pickle


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task()
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


@task()
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


@task()
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)

    logger.info(f"The MSE of validation is: {mse}")
    return


@task
def get_paths(date):
    logger = get_run_logger()
    base_url = "https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_"
    local_path = "./data/fhv_tripdata_"
    data_folder = "./data"
    file_format = ".parquet"
    date_format = "%Y-%m-%d"

    if date is None:
        rel_date = datetime.now()
    else:
        rel_date = datetime.strptime(date, date_format)

    training_datetime = rel_date + relativedelta(months=-2)
    validation_datetime = rel_date + relativedelta(months=-1)

    train_year, train_month = training_datetime.year, training_datetime.month
    val_year, val_month = validation_datetime.year, validation_datetime.month

    train_month = get_real_month(train_month)
    val_month = get_real_month(val_month)

    train_path = f'{local_path}{train_year}-{train_month}{file_format}'
    val_path = f'{local_path}{val_year}-{val_month}{file_format}'

    train_url = f'{base_url}{train_year}-{train_month}{file_format}'
    valid_url = f'{base_url}{val_year}-{val_month}{file_format}'
    logger.info(f"Going to Download data from: {train_url}, {valid_url}")
    train_filename = urllib.request.urlretrieve(train_url, train_path)
    valid_filename = urllib.request.urlretrieve(valid_url, val_path)

    return train_path, val_path


def get_real_month(month, max_val=10):
    if month < max_val:
        return f'0{month}'
    else:
        return f'{month}'


@flow()
def main(date=None):
    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, train=False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    with open(f'model-{date}.pkl', 'wb') as file:
        pickle.dump(lr, file)

    with open(f'dv-{date}.pkl', 'wb') as file:
        pickle.dump(dv, file)

    run_model(df_val_processed, categorical, dv, lr)


main(date="2021-08-15")
