# Analysis can be run locally

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

def collect_training_data(directory = ''):
    # 1. merge all csv in training data
    lst = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                lst.append(pd.read_csv(os.path.join(root, file)))

    df_raw = pd.concat(lst)
    # print(df_raw.head())
    # print(df_raw.columns)
    print(f'length starts: {len(df_raw)}')
    numeric_feature_lst = list(df_raw)
    for col in ['size', 'run_time', 'run_id', 'program', 'hostname']:
        numeric_feature_lst.remove(col)
    for col in numeric_feature_lst:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    df_raw.to_csv('training_data_all.csv', index=False)
    # df_raw[numeric_feature_lst].corr().to_csv('corr.csv')
    # sns.heatmap(df_raw[numeric_feature_lst])
    # plt.show()
    return df_raw

# 2. expand host spec in training data and export to

# 3. prediction
def preprocess_training(df_raw):
    feature_lst = ['branch-instructions', 'cache-misses', 'LLC-load-misses', 'context-switches', 'threads', 'host_cpu_idle', 'host_memused']
    df = df_raw[feature_lst + ['speed_up']]
    # for col in list(df):
    #     df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    print(f'length after dropping: {len(df)}')
    # imp = SimpleImputer(strategy="most_frequent")
    # df = imp.fit_transform(df)
    return df

def predict_regression(df):
    feature_lst = list(df)
    feature_lst.remove("speed_up")
    X = df[feature_lst]
    y = df['speed_up']

    # Split the DataFrame into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create a scikit-learn model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    print('model.score', model.score(X_test, y_test))
    print('MSRE', mean_squared_error(y_test, y_pred))


if __name__ == '__main__':
    df_raw = collect_training_data(r'C:\Users\yingl\OneDrive\Desktop\MultiorePerformancePrediction\MultiorePerformancePrediction\data\training_data')
    # df = preprocess_training(df_raw)
    # predict_regression(df)