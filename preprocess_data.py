import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,cross_val_score


def split_dataset(filename):
    df = pd.read_csv(filename)
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
    df = df.drop('id', axis=1)

    df_glucose = sorted(df['avg_glucose_level'])
    Q1, Q3 = np.percentile(df_glucose, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (3 * IQR)
    upper_range = Q3 + (3 * IQR)

    df = df.drop(df[df.avg_glucose_level > upper_range].index)

    df_bmi = sorted(df['bmi'])
    Q1, Q3 = np.percentile(df_bmi, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)

    df = df.drop(df[df.bmi > upper_range].index)

    df2 = df.copy()

    df2['age'] = df2['age'].apply(lambda x: np.log(x + 10) * 3)
    df2['avg_glucose_level'] = df2['avg_glucose_level'].apply(lambda x: np.log(x + 10) * 2)
    df2['bmi'] = df2['bmi'].apply(lambda x: np.log(x + 10) * 2)

    df2['gender'] = df2["gender"].map({"Male": 0, "Female": 1, "Other": 2}).astype(int)
    df2['ever_married'] = df2["ever_married"].map({"Yes": 1, "No": 0}).astype(int)
    df2['Residence_type'] = df2["Residence_type"].map({"Urban": 1, "Rural": 0}).astype(int)
    df2['work_type'] = df2['work_type'].map({"Private": 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3,
                                             "Never_worked": 4})
    df2['smoking_status'] = df2['smoking_status'].map({'never smoked': 0, 'Unknown': 1, 'formerly smoked': 2,
                                                       "smokes": 3})

    Y_new = df2['stroke']
    X_new = df2.drop('stroke', axis=1)

    x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(X_new, Y_new, test_size=0.2, random_state=42)

    return x_train_new, x_test_new, y_train_new, y_test_new


if __name__ == '__main__':
    res = split_dataset('./healthcare-dataset-stroke-data.csv')
    print(1)


