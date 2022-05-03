import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, make_scorer


data_path = "./"
train = pd.read_csv(data_path + 'data.csv')
feats = [f"x{i}" for i in range(1, 11)]
X, Y = train[feats], train["y"]

lm = Ridge(alpha=0).fit(X, Y)
imp_lm = permutation_importance(lm, X, Y, n_repeats=30, random_state=0)
print(*(imp_lm.importances_mean.argsort()[::-1] + 1))

mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", batch_size=256, early_stopping=True).fit(X.values, Y)
imp_mlp = permutation_importance(mlp, X.values, Y, n_repeats=30, random_state=0)
print(*(imp_mlp.importances_mean.argsort()[::-1] + 1))