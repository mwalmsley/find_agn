import numpy as np
a  = np.linspace(0., 1.)
b = np.linspace(1., 2.)
g = np.stack([a, b], axis=-1)
print(g.shape)
for x in np.nditer(g):
    print(x)


def multi_output_regression(df, x_cols, y_cols):
    # Wrap estimator for multi-output regression. Can only return one value per y_col, not e.g. std
    df = df.copy().dropna(subset=x_cols + y_cols, how='any')

    X = df[x_cols].values
    y = df[y_cols].values

    estimator = BayesianRidge()  # linear, not good in this case
    regressor = MultiOutputRegressor(estimator, n_jobs=-1)
    regressor.fit(X, y)

    return lambda x: regressor.predict(x)