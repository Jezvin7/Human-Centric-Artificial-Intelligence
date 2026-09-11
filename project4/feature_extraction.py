from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


APP_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET_PATH = (
    APP_DIR / "data" / "movie_metadata.csv"
)


NUMERIC_FEATURES = [
    "duration",
    "title_year",
    "imdb_score",
]


REQUIRED_COLUMNS = [
    "movie_title",
    "genres",
    "duration",
    "title_year",
    "imdb_score",
    "content_rating",
]


def load_movie_dataset(csv_path=DEFAULT_DATASET_PATH):

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"IMDB dataset not found at {csv_path}. "
            "Place the dataset in "
            "project4/data/movie_metadata.csv"
        )

    df = pd.read_csv(csv_path).copy()

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    # Remove movies without titles
    df = df.dropna(
        subset=["movie_title"]
    ).copy()

    # Clean text values
    df["movie_title"] = (
        df["movie_title"]
        .astype(str)
        .str.strip()
    )

    df["genres"] = (
        df["genres"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    df["content_rating"] = (
        df["content_rating"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    # Remove duplicate records
    df = df.drop_duplicates(
        subset=[
            "movie_title",
            "title_year"
        ]
    )

    df = df.reset_index(
        drop=True
    )

    # ID used internally by study
    df.insert(
        0,
        "movie_id",
        range(len(df))
    )

    return df


def extract_movie_features(
    csv_path=DEFAULT_DATASET_PATH
):

    df = load_movie_dataset(
        csv_path
    )

    # ------------------------------------------------
    # Genre multi-hot encoding
    # ------------------------------------------------

    genre_features = (
        df["genres"]
        .replace("", "Unknown")
        .str.get_dummies(sep="|")
    )

    genre_features.columns = [
        "genre_"
        + str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        for column in genre_features.columns
    ]

    genre_features = (
        genre_features.astype(float)
    )

    # ------------------------------------------------
    # Numerical features
    # ------------------------------------------------

    numeric = df[
        NUMERIC_FEATURES
    ].copy()

    for column in NUMERIC_FEATURES:

        numeric[column] = pd.to_numeric(
            numeric[column],
            errors="coerce"
        )

        median = (
            numeric[column]
            .median()
        )

        numeric[column] = (
            numeric[column]
            .fillna(median)
        )

    scaler = StandardScaler()

    numeric[
        NUMERIC_FEATURES
    ] = scaler.fit_transform(
        numeric[NUMERIC_FEATURES]
    )

    # ------------------------------------------------
    # Content rating
    # ------------------------------------------------

    rating_features = pd.get_dummies(
        df["content_rating"]
        .replace("", "Unknown"),
        prefix="rating",
        dtype=float
    )

    # ------------------------------------------------
    # Final x vector
    # ------------------------------------------------

    X = pd.concat(
        [
            genre_features,
            numeric.astype(float),
            rating_features
        ],
        axis=1
    )

    X = X.reset_index(
        drop=True
    )

    if X.isnull().any().any():
        raise ValueError(
            "Feature matrix contains missing values."
        )

    # ------------------------------------------------
    # Display information
    # ------------------------------------------------

    display_columns = [
        "movie_id",
        "movie_title",
        "genres",
        "duration",
        "title_year",
        "imdb_score",
        "content_rating",
    ]

    if "director_name" in df.columns:
        display_columns.append(
            "director_name"
        )

    movies = (
        df[display_columns]
        .copy()
        .reset_index(drop=True)
    )

    return (
        movies,
        X,
        list(X.columns),
        scaler,
    )


@lru_cache(maxsize=1)
def get_movie_data():
    """
    Load the dataset only once while
    the Django server is running.
    """

    return extract_movie_features(
        DEFAULT_DATASET_PATH
    )


def movie_as_dict(movie_id):

    movies, _, _, _ = (
        get_movie_data()
    )

    row = movies.loc[
        movies["movie_id"]
        == int(movie_id)
    ]

    if row.empty:
        raise KeyError(
            f"Unknown movie id {movie_id}"
        )

    item = (
        row.iloc[0]
        .to_dict()
    )

    # Replace NaN with readable text
    for key, value in list(
        item.items()
    ):
        if pd.isna(value):
            item[key] = "Unknown"

    return item