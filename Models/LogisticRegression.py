import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

data = pd.read_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/Processed Data/finalData.csv')
futureData = data[data['year'] == 2026].drop(['republican_victory'], axis=1)
data = data[data['year'] != 2026]

y = data['republican_victory'].astype(bool)
X = data.drop(['state_po', 'year', 'district', 'republican_victory'], axis = 1)
futureX = futureData.drop(['state_po', 'year', 'district',], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

pipe_lr = Pipeline([('std', StandardScaler()), ('lr', LogisticRegression())])

param_range = [0.01, 0.1, 1, 10, 100]

lr_param_grid = {
    'lr__C':param_range,
    'lr__solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'lr__max_iter':[100, 500, 1000]
}

lr_gs = GridSearchCV(estimator=pipe_lr, param_grid=lr_param_grid, scoring='f1', refit=True, cv=10, verbose=3)
    
lr_gs = lr_gs.fit(X_train, y_train)

print("Best Logistic Regression Model: ")
print("Model hyper-parameters: ", lr_gs.best_params_)
print("Validation data F1: ", lr_gs.best_score_)

print("Training Classification Report: ")
y_train_pred = lr_gs.best_estimator_.predict(X_train)
print(classification_report(y_train, y_train_pred))

print("Final Classification Report: ")
y_pred = lr_gs.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))

futureData['Republican Prob'] = lr_gs.best_estimator_.predict_proba(futureX)[:,1]
futureData.to_csv('C:/Users/finco/Documents/GitHub/DS496Capstone/2026 Predictions/FutureModelPredictionData.csv', index=False)