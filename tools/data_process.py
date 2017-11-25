import pandas as pd


def read_data(csv_file):
    df = pd.read_csv(csv_file)
    one_hot = pd.get_dummies(df['author'])
    df = df.drop('author', axis=1)
    df = df.join(one_hot)

    return df


def df_to_csv(df, output_name='data/one_hot_train'):
    df.to_csv(output_name)
    return df
