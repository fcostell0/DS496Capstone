from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

def RF(X_train,y_train):
    pipe_rf = Pipeline([('std', StandardScaler()), ('rf', RandomForestClassifier())])

    rf_param_grid = {
        'rf__n_estimators':[100, 500, 1000],
        'rf__criterion':['gini', 'entropy', 'log_loss'],
        'rf__max_features':['sqrt', 'log2']
    }

    RF_gs = GridSearchCV(estimator=pipe_rf, param_grid=rf_param_grid, scoring='f1', refit=True, cv=5, verbose=3)

    RF_gs = RF_gs.fit(X_train, y_train)

    best_RF = RF_gs.best_estimator_

    calibrated_rf = CalibratedClassifierCV(estimator=best_RF.named_steps['rf'], method='sigmoid', cv=3)
    calibrated_rf.fit(X_train, y_train)

    return calibrated_rf

