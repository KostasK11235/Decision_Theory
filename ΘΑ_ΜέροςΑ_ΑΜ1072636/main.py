import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score

data = pd.read_csv('Indian Liver Patient Datase (ILPD).csv')

# copying the dataframe to a new csv to normalise the data
data.to_csv('new_data.csv', index=False)

Age_sum, TB_sum, DB_sum, AAP_sum, SgptAA_sum, SgotAA_sum, TP_sum, ALB_sum, AGR_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0

# adding values in each column
for i in range(0, data.shape[0]):
    Age_sum += data.loc[i, 'Age']
    TB_sum += data.loc[i, 'TB']
    DB_sum += data.loc[i, 'DB']
    AAP_sum += data.loc[i, 'AAP']
    SgptAA_sum += data.loc[i, 'Sgpt']
    SgotAA_sum += data.loc[i, 'Sgot']
    TP_sum += data.loc[i, 'TP']
    ALB_sum += data.loc[i, 'ALB']
    AGR_sum += data.loc[i, 'AGR']

# calculating expected value
Age_exp = round(Age_sum/data.shape[0], 4)
TB_exp = round(TB_sum/data.shape[0], 4)
DB_exp = round(DB_sum/data.shape[0], 4)
AAP_exp = round(AAP_sum/data.shape[0], 4)
Sgpt_exp = round(SgptAA_sum/data.shape[0], 4)
Sgot_exp = round(SgotAA_sum/data.shape[0], 4)
TP_exp = round(TP_sum/data.shape[0], 4)
ALB_exp = round(ALB_sum/data.shape[0], 4)
AGR_exp = round(AGR_sum/data.shape[0], 4)

# finding min-max from each column to calculate d = max - min
maxValues = data[['Age', 'TB', 'DB', 'AAP', 'Sgpt', 'Sgot', 'TP', 'ALB', 'AGR']].max()
minValues = data[['Age', 'TB', 'DB', 'AAP', 'Sgpt', 'Sgot', 'TP', 'ALB', 'AGR']].min()

# calculating d for each column
Age_d = int(maxValues[0] - minValues[0])
TB_d = float(maxValues[1] - minValues[1])
DB_d = float(maxValues[2] - minValues[2])
AAP_d = int(maxValues[3] - minValues[3])
Sgpt_d = int(maxValues[4] - minValues[4])
Sgot_d = int(maxValues[5] - minValues[5])
TP_d = float(maxValues[6] - minValues[6])
ALB_d = float(maxValues[7] - minValues[7])
AGR_d = float(maxValues[8] - minValues[8])

# updating the new_data.csv file with the normalised data
patients = pd.read_csv('new_data.csv')

for i in range(0, patients.shape[0]):
    patients.loc[i, 'Age'] = round((patients.loc[i, 'Age']-Age_exp)/Age_d, 4)
    patients.loc[i, 'TB'] = round((patients.loc[i, 'TB']-TB_exp)/TB_d, 4)
    patients.loc[i, 'DB'] = round((patients.loc[i, 'DB'] - DB_exp) / DB_d, 4)
    patients.loc[i, 'AAP'] = round((patients.loc[i, 'AAP'] - AAP_exp) / AAP_d, 4)
    patients.loc[i, 'Sgpt'] = round((patients.loc[i, 'Sgpt'] - Sgpt_exp) / Sgpt_d, 4)
    patients.loc[i, 'Sgot'] = round((patients.loc[i, 'Sgot'] - Sgot_exp) / Sgot_d, 4)
    patients.loc[i, 'TP'] = round((patients.loc[i, 'TP'] - TP_exp) / TP_d, 4)
    patients.loc[i, 'ALB'] = round((patients.loc[i, 'ALB'] - ALB_exp) / ALB_d, 4)
    patients.loc[i, 'AGR'] = round((patients.loc[i, 'AGR'] - AGR_exp) / AGR_d, 4)

patients.to_csv('new_data.csv', index=False)

print(patients.head())

# dataframe with the data except the Dataset column
X = patients.drop(['Gender', 'Dataset'], axis=1)
# dataframe with the target values only
y = patients['Dataset']

# training and testing the Naive Bayes with 5-fold cross validation
print("---------------------------------------------------------")
gaussian = GaussianNB()
g_scores = cross_val_score(gaussian, X, y, cv=5)
print("5-Fold cross validation scores (Gaussian Naive Bayes): \n", g_scores)
print("%0.2f accuracy with standard deviation of %0.2f" % (g_scores.mean(), g_scores.std()))
print("---------------------------------------------------------")
scorer = make_scorer(specificity_score, greater_is_better=True)
g_scores = cross_val_score(gaussian, X, y, cv=5, scoring=scorer)
print("Specificity scores: \n", g_scores)
specificity = g_scores.mean()
print("%0.2f accuracy with standard deviation of %0.2f" % (g_scores.mean(), g_scores.std()))
print("---------------------------------------------------------")
scorer = make_scorer(sensitivity_score, greater_is_better=True)
g_scores = cross_val_score(gaussian, X, y, cv=5, scoring=scorer)
print("Sensitivity scores: \n", g_scores)
sensitivity = g_scores.mean()
print("%0.2f accuracy with standard deviation of %0.2f" % (g_scores.mean(), g_scores.std()))
print("---------------------------------------------------------")
scorer = make_scorer(geometric_mean_score, greater_is_better=True)
g_scores = cross_val_score(gaussian, X, y, cv=5, scoring=scorer)
print("Geometric mean scores: \n", g_scores)
geometricMean = g_scores.mean()
print("%0.2f accuracy with standard deviation of %0.2f" % (g_scores.mean(), g_scores.std()))

