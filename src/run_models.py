import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


def print_evaluation_metrics(trained_model, trained_model_name,X_train, y_train,X_test, y_test):
    print("--------- For Model: ", trained_model_name, " ---------\n")
    training_preds = trained_model.predict(X_train)
    test_preds = trained_model.predict(X_test)

    print("-----Training Set ------")
    print("Mean squared error: ", mean_squared_error(y_train, training_preds))
    print("R2: ", r2_score(y_train, training_preds))

    print("\n")

    print("-----Test Set ------")
    print("Mean squared error: ", mean_squared_error(y_test, test_preds))
    print("R2: ", r2_score(y_test, test_preds))
    print("\n")



def LinearModel(X_train, y_train, X_test, y_test):

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print_evaluation_metrics(lr, "OLS Model",X_train, y_train, X_test, y_test)


def LinearModelRidge(X_train, y_train, X_test, y_test):

    alphas = 10 ** np.linspace(10, -2, 100) * 0.5
    ridgecv = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error", cv=10)
    ridgecv.fit(X_train, y_train)

    print("Value of lambda ", ridgecv.alpha_)

    ridge = Ridge()
    ridge.set_params(alpha=ridgecv.alpha_)
    ridge.fit(X_train, y_train)

    print_evaluation_metrics(ridge, "Ridge Model", X_train, y_train,X_test, y_test)


def LinearModelLasso(X_train, y_train, X_test, y_test):

    alphas = 10 ** np.linspace(10, -2, 100) * 0.5
    lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=100000, cv=10)
    lasso_cv.fit(X_train, y_train)

    print("Value of lasso tuning parameter", lasso_cv.alpha_)
    lasso = Lasso()

    lasso.set_params(alpha=lasso_cv.alpha_)
    lasso.fit(X_train, y_train)

    print_evaluation_metrics(lasso, "Lasso Model",X_train, y_train, X_test, y_test)

    print("\n")
    print("--- Feature Selection -----")

    coef = pd.Series(lasso.coef_, index=X_train.columns)

    print(
        "Lasso picked "
        + str(sum(coef != 0))
        + " variables and eliminated the other "
        + str(sum(coef == 0))
        + " variables"
    )

    for e in sorted(list(zip(list(X_train), lasso.coef_)), key=lambda e: -abs(e[1])):
        if e[1] != 0:
            print("\t{}, {:.3f}".format(e[0], e[1]))


def DecisionTreeModel(X_train, y_train, X_test, y_test):
    tree = DecisionTreeRegressor(random_state=123)
    tree.fit(X_train, y_train)
    print_evaluation_metrics(tree, "Decision Tree Model", X_train, y_train, X_test,y_test)

def XGBoostModel(X_train, y_train, X_test, y_test):
    xgb_reg = xgb.XGBRegressor()

    xgb_reg.fit(X_train, y_train)

    ft_weights_xgb_reg = pd.DataFrame(xgb_reg.feature_importances_, columns=['weight'], index=X_train.columns)

    print_evaluation_metrics(xgb_reg, "XGBoost Model", X_train, y_train, X_test, y_test)

    print("--------Feature Importance------------\n")
    print("The top 20  most important features are:")
    # print(ft_weights_xgb_reg.sort_values('weight')[:20])
    print(ft_weights_xgb_reg.sort_values('weight', ascending=False)[:20])
    print(ft_weights_xgb_reg.sort_values('weight', ascending=False)[:20].to_latex())


    # print("--------Feature Importance Plot------------\n")
    # plt.figure(figsize=(20,15))
    # plt.barh(ft_weights_xgb_reg.index, ft_weights_xgb_reg.weight, align='center')
    # plt.title("Feature importances in the XGBoost model", fontsize=14)
    # plt.xlabel("Feature importance")
    # plt.margins(y=0.01)
    # plt.tight_layout()
    # plt.show()


# def print_vif():




if __name__ == "__main__":

    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")

    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    LinearModel(X_train, y_train, X_test, y_test)
    LinearModelRidge(X_train, y_train, X_test, y_test)
    LinearModelLasso(X_train, y_train, X_test, y_test)
    # DecisionTreeModel(X_train, y_train, X_test, y_test)
    # XGBoostModel(X_train, y_train, X_test, y_test)
