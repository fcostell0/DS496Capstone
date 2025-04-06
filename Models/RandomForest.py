import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

#educationData = pd.read_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/compressedEducationData.csv')
#raceData = pd.read_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/raceData.csv')
#labels = pd.read_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/electionLabels.csv')

#data = pd.merge(educationData, raceData, how='inner', on=['state_po', 'year', 'district'])
#data = pd.merge(data, labels, how='inner', on=['state_po', 'year', 'district'])

data = pd.read_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/finalData.csv')


y = data['republican_victory']
X = data.drop(['state_po', 'year', 'district', 'republican_victory'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

pipe_rf = Pipeline([('std', StandardScaler()), ('rf', RandomForestClassifier())])
rf_param_grid = {
    'rf__n_estimators':[100, 500, 1000],
    'rf__criterion':['gini', 'entropy', 'log_loss'],
    'rf__max_features':['sqrt', 'log2']
}

RF_gs = GridSearchCV(estimator=pipe_rf, param_grid=rf_param_grid, scoring='f1', refit=True, cv=10, verbose=3)

RF_gs = RF_gs.fit(X_train, y_train)

print("Best Random Forest Model: ")
print("Model hyper-parameters: ", RF_gs.best_params_)
print("Training data F1 score: ", RF_gs.best_score_)
