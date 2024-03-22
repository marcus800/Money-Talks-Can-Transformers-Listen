
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

#Train a softmax regression just using the last 3 predictions, nothing else
#Also saving the training dist

#Next steps could be training more complex models, maybe including other fetures or ect ect.

data_types = ["S&P","FX"]
for data_type in data_types:
    csv_location = f"train_data/{data_type}/data.csv"
    df = pd.read_csv(csv_location)
    price_change_map = {
        "MAJOR_INCREASE": 2,
        "MINOR_INCREASE": 1,
        "NO_CHANGE": 0,
        "MINOR_DECREASE": -1,
        "MAJOR_DECREASE": -2
    }
    df['label_mapped'] = df['label'].map(price_change_map)

    features = df['label_mapped'].shift(-1).fillna(method='ffill').to_frame()
    features = features.rename(columns={'label_mapped': 'shift_1'})
    features['shift_2'] = df['label_mapped'].shift(-2).fillna(method='ffill')
    features['shift_3'] = df['label_mapped'].shift(-3).fillna(method='ffill')
    target = df['label_mapped']

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(features, target)

    model_filename = f"../saved_models/{data_type}/softmax_regression_model.pkl"
    joblib.dump(model, model_filename)

    label_distribution = df['label'].value_counts(normalize=True).to_dict()
    label_dist_filename = f"../saved_models/{data_type}/label_distribution.pkl"
    joblib.dump(label_distribution, label_dist_filename)
