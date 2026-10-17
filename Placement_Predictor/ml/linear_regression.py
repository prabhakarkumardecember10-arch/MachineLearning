from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ml.data_loader import load_data
from ml.preprocessing import preprocess_data


def train_model():

    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        feature_names
    ) = preprocess_data(df)

    model = LinearRegression()

    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, mae, r2