import random

from .feature_extraction import (
    get_movie_data
)


PAIRWISE_ROUNDS = 15

RANKING_ROUNDS = 3

RANKING_SIZE = 10

EVALUATION_ROUNDS = 10


def _sample_unique_movie_ids(
    total_required
):

    movies, _, _, _ = (
        get_movie_data()
    )

    ids = (
        movies["movie_id"]
        .astype(int)
        .tolist()
    )

    if len(ids) < total_required:

        raise ValueError(
            "Not enough movies in dataset."
        )

    return random.sample(
        ids,
        total_required
    )


def build_study_items():

    # Pairwise:
    # 15 comparisons x 2 movies = 30 movies
    #
    # Ranking:
    # 3 rankings x 10 movies = 30 movies
    #
    # Evaluation:
    # 10 pairs x 2 = 20 new movies

    total = (
        PAIRWISE_ROUNDS * 2
        + RANKING_ROUNDS * RANKING_SIZE
        + EVALUATION_ROUNDS * 2
    )

    ids = (
        _sample_unique_movie_ids(
            total
        )
    )

    cursor = 0

    pairwise_pairs = []

    for _ in range(
        PAIRWISE_ROUNDS
    ):

        pairwise_pairs.append(
            [
                ids[cursor],
                ids[cursor + 1]
            ]
        )

        cursor += 2


    ranking_sets = []

    for _ in range(
        RANKING_ROUNDS
    ):

        ranking_sets.append(
            ids[
                cursor:
                cursor + RANKING_SIZE
            ]
        )

        cursor += RANKING_SIZE


    evaluation_pairs = []

    for _ in range(
        EVALUATION_ROUNDS
    ):

        evaluation_pairs.append(
            [
                ids[cursor],
                ids[cursor + 1]
            ]
        )

        cursor += 2


    return (
        pairwise_pairs,
        ranking_sets,
        evaluation_pairs
    )