import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_shape(df):
    return df.shape

def get_columns(df):
    return df.columns.tolist()

def get_dtypes(df):
    return df.dtypes

def get_missing_values(df):
    return df.isna().sum()

def get_duplicates(df):
    return df.duplicated().sum()

def descriptive_statistics(df):
    numeric_df = df.select_dtypes(include="number")
    stats = pd.DataFrame(index=numeric_df.columns)

    stats["Data Type"] = numeric_df.dtypes
    stats["Mean"] = numeric_df.mean()
    stats["Median"] = numeric_df.median()
    stats["Mode"] = numeric_df.mode().iloc[0]
    stats["Standard Deviation"] = numeric_df.std()
    stats["Variance"] = numeric_df.var()
    stats["Minimum"] = numeric_df.min()
    stats["25%"] = numeric_df.quantile(0.25)
    stats["75%"] = numeric_df.quantile(0.75)
    stats["Maximum"] = numeric_df.max()
    stats["Range"] = numeric_df.max() - numeric_df.min()
    stats["Missing Values"] = numeric_df.isna().sum()
    stats["Unique Values"] = numeric_df.nunique()

    return stats.round(2)

def correlation_matrix(df):
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.corr()

def plot_heatmap(df):
    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(18, 14))

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar=True
    )

    plt.title("Correlation Heatmap", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    plt.savefig(
        "static/plots/heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plot_histogram(df, column):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True)
    plt.title(f"{column} Distribution")
    plt.tight_layout()
    plt.savefig(f"static/plots/{column}_histogram.png")
    plt.close()

def plot_boxplot(df, column):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[column])
    plt.title(f"{column} Box Plot")
    plt.tight_layout()
    plt.savefig(f"static/plots/{column}_boxplot.png")
    plt.close()

def plot_countplot(df, column):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"static/plots/{column}_countplot.png")
    plt.close()

def plot_scatter(df, x_column, y_column):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_column, y=y_column)
    plt.tight_layout()
    plt.savefig(f"static/plots/{x_column}_{y_column}_scatter.png")
    plt.close()