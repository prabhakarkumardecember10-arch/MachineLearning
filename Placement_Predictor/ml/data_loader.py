from pathlib import Path
import pandas as pd


def load_data():
    project_root = Path(__file__).resolve().parent.parent

    file_path = project_root / "data" / "placement_predict_50k Dataset.csv"

    df = pd.read_csv(file_path)

    return df