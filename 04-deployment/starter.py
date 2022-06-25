import os
import sys
import pickle
import pandas as pd


def load_model(path, perm='rb'):
    print(f"Going to load model from {path}...")
    with open(path, perm) as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr


def read_data(filename, categorical):
    print("Going to load data from:", filename)
    df = pd.read_parquet(filename)
    df['ride_id'] = '{year:04d}/{month:02d}_' + df.index.astype('str')
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def apply_model(model_path, input_file, output_file, categorical):
    df = read_data(input_file, categorical=categorical)
    dicts = df[categorical].to_dict(orient='records')
    dv, lr = load_model(model_path)
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print("Mean predictions:", y_pred.mean())

    # prepare a new dataframe and save it for batch
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    # save to parquet
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    print("Output size:", round(
        os.path.getsize(output_file) / 1024 / 1024, 3))


def run():

    categorical = ['PUlocationID', 'DOlocationID']
    year = int(sys.argv[0])
    month = int(sys.argv[2])
    model_path = sys.argv[3]
    output_file = f"predictions_df_fhv_tripdata_{year:04d}-{month:02d}.parquet"
    input_file = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'

    apply_model(
        model_path=model_path,
        input_file=input_file,
        output_file=output_file,
        categorical=categorical
    )


if __name__ == '__main__':
    run()
