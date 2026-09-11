import numpy as np

from scipy.optimize import minimize
from scipy.special import expit, logsumexp


DEFAULT_L2 = 1.0


def _as_numpy(X):

    return np.asarray(
        X,
        dtype=float
    )


# =====================================================
# Bradley-Terry
# =====================================================

def fit_bradley_terry(
    pairwise_responses,
    X,
    l2=DEFAULT_L2
):

    X = _as_numpy(X)

    number_of_features = (
        X.shape[1]
    )

    if not pairwise_responses:
        return np.zeros(
            number_of_features
        )

    def objective(w):

        loss = 0.0

        for response in pairwise_responses:

            left = int(
                response["left"]
            )

            right = int(
                response["right"]
            )

            chosen = int(
                response["chosen"]
            )

            if chosen == left:

                difference = (
                    X[left] - X[right]
                )

            elif chosen == right:

                difference = (
                    X[right] - X[left]
                )

            else:

                raise ValueError(
                    "Chosen movie was not presented."
                )

            margin = float(
                np.dot(
                    w,
                    difference
                )
            )

            # Negative log likelihood
            loss += np.logaddexp(
                0.0,
                -margin
            )

        # L2 regularization
        loss += (
            0.5
            * l2
            * float(
                np.dot(w, w)
            )
        )

        return loss

    result = minimize(
        objective,
        np.zeros(number_of_features),
        method="L-BFGS-B"
    )

    if not result.success:
        raise RuntimeError(
            "Bradley-Terry optimization failed: "
            + result.message
        )

    return result.x


# =====================================================
# Plackett-Luce
# =====================================================

def fit_plackett_luce(
    ranking_responses,
    X,
    l2=DEFAULT_L2
):

    X = _as_numpy(X)

    number_of_features = (
        X.shape[1]
    )

    if not ranking_responses:
        return np.zeros(
            number_of_features
        )

    def objective(w):

        loss = 0.0

        for response in ranking_responses:

            ranking = [
                int(movie_id)
                for movie_id
                in response["ranking"]
            ]

            features = X[
                ranking
            ]

            utilities = (
                features @ w
            )

            for k in range(
                len(ranking) - 1
            ):

                loss += (
                    -utilities[k]
                    + logsumexp(
                        utilities[k:]
                    )
                )

        loss += (
            0.5
            * l2
            * float(
                np.dot(w, w)
            )
        )

        return float(loss)

    result = minimize(
        objective,
        np.zeros(number_of_features),
        method="L-BFGS-B"
    )

    if not result.success:
        raise RuntimeError(
            "Plackett-Luce optimization failed: "
            + result.message
        )

    return result.x


# =====================================================
# Prediction
# =====================================================

def pairwise_probability(
    weights,
    X,
    movie_i,
    movie_j
):

    X = _as_numpy(X)

    w = np.asarray(
        weights,
        dtype=float
    )

    difference = (
        X[int(movie_i)]
        - X[int(movie_j)]
    )

    margin = float(
        np.dot(
            w,
            difference
        )
    )

    return float(
        expit(margin)
    )


# =====================================================
# Evaluation
# =====================================================

def evaluate_on_pairwise_choices(
    weights,
    evaluation_responses,
    X
):

    if not evaluation_responses:

        return {
            "accuracy": None,
            "log_loss": None,
            "n": 0
        }

    correct = 0

    losses = []

    epsilon = 1e-12

    for response in evaluation_responses:

        left = int(
            response["left"]
        )

        right = int(
            response["right"]
        )

        chosen = int(
            response["chosen"]
        )

        probability_left = (
            pairwise_probability(
                weights,
                X,
                left,
                right
            )
        )

        predicted = (
            left
            if probability_left >= 0.5
            else right
        )

        if predicted == chosen:
            correct += 1

        if chosen == left:
            probability_chosen = (
                probability_left
            )
        else:
            probability_chosen = (
                1.0
                - probability_left
            )

        losses.append(
            -np.log(
                np.clip(
                    probability_chosen,
                    epsilon,
                    1.0
                )
            )
        )

    return {
        "accuracy":
            float(
                correct
                / len(
                    evaluation_responses
                )
            ),

        "log_loss":
            float(
                np.mean(losses)
            ),

        "n":
            len(
                evaluation_responses
            ),
    }