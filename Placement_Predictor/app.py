from flask import Flask, render_template
from ml.linear_regression import train_model
from ml.data_loader import load_data
from ml.preprocessing import get_preprocessing_summary
from ml.eda import (
    get_shape,
    get_columns,
    get_dtypes,
    get_missing_values,
    get_duplicates,
    descriptive_statistics,
    plot_heatmap,
    plot_histogram,
    plot_boxplot,
    plot_countplot,
    plot_scatter
)

from ml.preprocessing import get_preprocessing_summary


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/data_loading")
def data_loading():

    df = load_data()

    return render_template(
        "data_loading.html",
        rows=df.shape[0],
        columns=df.shape[1],
        column_names=df.columns.tolist(),
        preview=df.head().to_html(
            classes="table table-striped",
            index=False
        )
    )


@app.route("/eda")
def eda():

    df = load_data()

    shape = get_shape(df)

    plot_heatmap(df)
    plot_histogram(df, "CGPA")
    plot_boxplot(df, "CGPA")
    plot_countplot(df, "PlacementStatus")
    plot_scatter(df, "CGPA", "CodingTestScore")

    return render_template(
        "eda.html",
        rows=shape[0],
        columns_count=shape[1],
        duplicates=get_duplicates(df),
        column_names=get_columns(df),
        dtypes=get_dtypes(df).to_frame(
            name="Data Type"
        ).to_html(
            classes="table",
            index=True
        ),
        missing_values=get_missing_values(df).to_frame(
            name="Missing Values"
        ).to_html(
            classes="table",
            index=True
        ),
        statistics=descriptive_statistics(df).to_html(
            classes="table table-striped",
            index=True
        )
    )


@app.route("/preprocessing")
def preprocessing():

    df = load_data()

    summary = get_preprocessing_summary(df)

    return render_template(
        "preprocessing.html",
        original_rows=summary["original_rows"],
        original_columns=summary["original_columns"],
        train_rows=summary["train_rows"],
        test_rows=summary["test_rows"],
        original_features=summary["original_features"],
        processed_features=summary["processed_features"],
        numeric_count=summary["numeric_count"],
        categorical_count=summary["categorical_count"],
        ordinal_count=summary["ordinal_count"],
        removed_features=summary["removed_features"],
        target=summary["target"],
        numeric_features=summary["numeric_features"],
        categorical_features=summary["categorical_features"],
        ordinal_features=summary["ordinal_features"],
        feature_names=summary["feature_names"],
        sample_original=summary["sample_original"],
        processed_sample=summary["processed_sample"]
    )

@app.route("/linear_regression")
def linear_regression():

    mse, rmse, mae, r2 = train_model()

    return render_template(
        "linear_regression.html",
        mse=round(mse, 4),
        rmse=round(rmse, 4),
        mae=round(mae, 4),
        r2=round(r2, 4),
        r2_percent=round(r2 * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)