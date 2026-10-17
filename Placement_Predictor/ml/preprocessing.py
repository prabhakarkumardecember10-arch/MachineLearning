from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from ml.data_loader import load_data

numeric_features = [
    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8",
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "ExtraCurricular"
]

categorical_features = [
    "Gender",
    "City",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs"
]

ordinal_features = [
    "CollegeTier",
    "CGPA_Tier"
]

removed_features = [
    "StudentID",
    "IsAnomaly",
    "Salary Package"
]


def split_data(df):
    X = df.drop(
        columns=[
            "PlacementStatus",
            "StudentID",
            "IsAnomaly",
            "Salary Package"
        ]
    )

    y = df["PlacementStatus"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[
                ["Tier1", "Tier2", "Tier3"],
                ["Low", "Mid", "High"]
            ]
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("ordinal", ordinal_pipeline, ordinal_features)
    ])

    return preprocessor
def get_preprocessing_summary(df):

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        feature_names
    ) = preprocess_data(df)

    sample_original = X_train.head(1).copy()

    sample_processed = X_train_processed[0].flatten()

    processed_sample = []

    for feature, value in zip(feature_names, sample_processed):
        processed_sample.append({
            "feature": feature,
            "value": round(float(value), 3)
        })

    return {
        "original_rows": df.shape[0],
        "original_columns": df.shape[1],
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "original_features": X_train.shape[1],
        "processed_features": X_train_processed.shape[1],
        "numeric_count": len(numeric_features),
        "categorical_count": len(categorical_features),
        "ordinal_count": len(ordinal_features),
        "removed_features": removed_features,
        "target": "PlacementStatus",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "ordinal_features": ordinal_features,
        "feature_names": feature_names.tolist(),
        "sample_original": sample_original.to_dict("records"),
        "processed_sample": processed_sample
    }

def preprocess_data(df):

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = create_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        feature_names
    )




def processed_processed_sample(sample):

    result = []

    for feature, value in sample.items():
        result.append({
            "feature": feature,
            "value": round(float(value), 3)
        })

    return result