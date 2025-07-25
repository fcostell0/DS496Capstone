from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

def LogReg(X_train,y_train):
    ### Pipeline/grid search
    pipe_lr = Pipeline([('std', StandardScaler()), ('lr', LogisticRegression())])

    param_range = [0.01, 0.1, 1, 10, 100]
    lr_param_grid = {
        'lr__C':param_range,
        'lr__solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        'lr__max_iter':[1000]
    }

    lr_gs = GridSearchCV(estimator=pipe_lr, param_grid=lr_param_grid, scoring='f1', refit=True, cv=5, verbose=3)
        
    lr_gs = lr_gs.fit(X_train, y_train)

    best_lr = lr_gs.best_estimator_

    calibrated_lr = CalibratedClassifierCV(estimator=best_lr.named_steps['lr'], method='sigmoid', cv=3)
    calibrated_lr.fit(X_train, y_train)

    return calibrated_lr
