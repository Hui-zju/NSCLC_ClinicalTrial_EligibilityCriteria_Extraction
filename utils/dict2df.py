import pandas as pd
import json


def dict2df(path):
    with open(path, 'r') as f:
        metrics = json.load(f)
    df = pd.DataFrame(metrics)
    df = df.T
    # df[['true']] = df[['true']].astype(int)
    # df[['predict']] = df[['predict']].astype(int)
    # df[['correct']] = df[['correct']].astype(int)
    return df


path = '/home/chenh/PycharmProjects/clinicial_trial_variation_extraction/algorithm/entity_recognition/lightning_logs/version_1/epoch_25_eval_test.json'
df = dict2df(path)
print(df)
