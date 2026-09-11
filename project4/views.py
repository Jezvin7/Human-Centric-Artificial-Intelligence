import random
import time

from django.http import Http404
from django.shortcuts import (
    redirect,
    render
)

from django.views.decorators.http import (
    require_http_methods
)

from .feature_extraction import (
    get_movie_data,
    movie_as_dict
)

from .models import StudySession

from .preference_model import (
    evaluate_on_pairwise_choices,
    fit_bradley_terry,
    fit_plackett_luce
)

from .study_utils import (
    EVALUATION_ROUNDS,
    PAIRWISE_ROUNDS,
    RANKING_ROUNDS,
    build_study_items
)


def index(request):

    return render(
        request,
        "project4/index.html"
    )


def _get_study(request):

    study_id = (
        request.session.get(
            "project4_study_id"
        )
    )

    if not study_id:
        return None

    return (
        StudySession.objects
        .filter(id=study_id)
        .first()
    )


def _create_study(request):

    (
        pairwise_pairs,
        ranking_sets,
        evaluation_pairs
    ) = build_study_items()

    study = (
        StudySession.objects.create(

            condition_order=random.choice(
                [
                    "pairwise_first",
                    "ranking_first"
                ]
            ),

            consent_given=True,

            current_stage="condition_1",

            pairwise_pairs=pairwise_pairs,

            ranking_sets=ranking_sets,

            evaluation_pairs=evaluation_pairs
        )
    )

    request.session[
        "project4_study_id"
    ] = study.id

    request.session[
        "project4_pairwise_started"
    ] = None

    request.session[
        "project4_ranking_started"
    ] = None

    return study


def _next_after_condition(
    study,
    finished_condition
):

    if (
        study.condition_order
        == "pairwise_first"
    ):

        if finished_condition == "pairwise":
            return "ranking"

        return "evaluation"

    else:

        if finished_condition == "ranking":
            return "pairwise"

        return "evaluation"


@require_http_methods(
    ["GET", "POST"]
)
def study_intro(request):

    dataset_error = None

    try:
        get_movie_data()

    except Exception as error:
        dataset_error = str(error)

    if request.method == "POST":

        if dataset_error:

            return render(
                request,
                "project4/study_intro.html",
                {
                    "dataset_error":
                        dataset_error
                }
            )

        if (
            request.POST.get(
                "consent"
            )
            != "yes"
        ):

            return render(
                request,
                "project4/study_intro.html",
                {
                    "error":
                        "Please provide consent before starting."
                }
            )

        old_study = (
            _get_study(request)
        )

        if (
            old_study
            and not old_study.completed
        ):
            old_study.delete()

        study = (
            _create_study(request)
        )

        if (
            study.condition_order
            == "pairwise_first"
        ):

            return redirect(
                "project4:pairwise"
            )

        return redirect(
            "project4:ranking"
        )

    return render(
        request,
        "project4/study_intro.html",
        {
            "dataset_error":
                dataset_error
        }
    )


@require_http_methods(
    ["GET", "POST"]
)
def pairwise(request):

    study = _get_study(
        request
    )

    if not study:

        return redirect(
            "project4:study_intro"
        )

    round_index = len(
        study.pairwise_responses
    )

    if (
        round_index
        >= PAIRWISE_ROUNDS
    ):

        return redirect(
            "project4:questionnaire",
            condition="pairwise"
        )

    if (
        request.session.get(
            "project4_pairwise_started"
        )
        is None
    ):

        request.session[
            "project4_pairwise_started"
        ] = time.time()

    left_id, right_id = (
        study.pairwise_pairs[
            round_index
        ]
    )

    if request.method == "POST":

        try:

            chosen = int(
                request.POST["chosen"]
            )

        except (
            KeyError,
            ValueError
        ):

            raise Http404(
                "Invalid response"
            )

        if chosen not in (
            int(left_id),
            int(right_id)
        ):

            raise Http404(
                "Chosen movie was not presented."
            )

        responses = list(
            study.pairwise_responses
        )

        responses.append(
            {
                "round":
                    round_index + 1,

                "left":
                    int(left_id),

                "right":
                    int(right_id),

                "chosen":
                    chosen
            }
        )

        study.pairwise_responses = (
            responses
        )

        study.save()

        return redirect(
            "project4:pairwise"
        )

    return render(
        request,
        "project4/pairwise.html",
        {
            "left":
                movie_as_dict(
                    left_id
                ),

            "right":
                movie_as_dict(
                    right_id
                ),

            "round_number":
                round_index + 1,

            "total_rounds":
                PAIRWISE_ROUNDS,

            "progress":
                round(
                    (
                        round_index
                        / PAIRWISE_ROUNDS
                    )
                    * 100
                )
        }
    )


@require_http_methods(
    ["GET", "POST"]
)
def ranking(request):

    study = _get_study(
        request
    )

    if not study:

        return redirect(
            "project4:study_intro"
        )

    round_index = len(
        study.ranking_responses
    )

    if (
        round_index
        >= RANKING_ROUNDS
    ):

        return redirect(
            "project4:questionnaire",
            condition="ranking"
        )

    if (
        request.session.get(
            "project4_ranking_started"
        )
        is None
    ):

        request.session[
            "project4_ranking_started"
        ] = time.time()

    current_ids = [
        int(movie_id)
        for movie_id
        in study.ranking_sets[
            round_index
        ]
    ]

    if request.method == "POST":

        raw = (
            request.POST.get(
                "ranking",
                ""
            )
        )

        try:

            ranking_ids = [
                int(value)
                for value
                in raw.split(",")
                if value != ""
            ]

        except ValueError:

            ranking_ids = []

        if (
            len(ranking_ids)
            != len(current_ids)
            or
            set(ranking_ids)
            != set(current_ids)
        ):

            return render(
                request,
                "project4/ranking.html",
                {
                    "movies": [
                        movie_as_dict(i)
                        for i
                        in current_ids
                    ],

                    "round_number":
                        round_index + 1,

                    "total_rounds":
                        RANKING_ROUNDS,

                    "progress":
                        round(
                            (
                                round_index
                                / RANKING_ROUNDS
                            )
                            * 100
                        ),

                    "error":
                        "Please rank all ten movies."
                }
            )

        responses = list(
            study.ranking_responses
        )

        responses.append(
            {
                "round":
                    round_index + 1,

                "ranking":
                    ranking_ids
            }
        )

        study.ranking_responses = (
            responses
        )

        study.save()

        return redirect(
            "project4:ranking"
        )

    return render(
        request,
        "project4/ranking.html",
        {
            "movies": [
                movie_as_dict(i)
                for i
                in current_ids
            ],

            "round_number":
                round_index + 1,

            "total_rounds":
                RANKING_ROUNDS,

            "progress":
                round(
                    (
                        round_index
                        / RANKING_ROUNDS
                    )
                    * 100
                )
        }
    )


@require_http_methods(
    ["GET", "POST"]
)
def questionnaire(
    request,
    condition
):

    if condition not in {
        "pairwise",
        "ranking"
    }:

        raise Http404(
            "Unknown condition"
        )

    study = _get_study(
        request
    )

    if not study:

        return redirect(
            "project4:study_intro"
        )

    if (
        condition == "pairwise"
        and
        len(
            study.pairwise_responses
        )
        < PAIRWISE_ROUNDS
    ):

        return redirect(
            "project4:pairwise"
        )

    if (
        condition == "ranking"
        and
        len(
            study.ranking_responses
        )
        < RANKING_ROUNDS
    ):

        return redirect(
            "project4:ranking"
        )

    key = (
        f"{condition}_questionnaire"
    )

    if (
        key
        in study.questionnaire_responses
    ):

        next_condition = (
            _next_after_condition(
                study,
                condition
            )
        )

        return redirect(
            f"project4:{next_condition}"
        )

    if request.method == "POST":

        def score(name):

            value = int(
                request.POST.get(
                    name,
                    0
                )
            )

            if value not in range(
                1,
                6
            ):
                raise ValueError

            return value

        try:

            answers = {

                "easy_to_express":
                    score(
                        "easy_to_express"
                    ),

                "mentally_demanding":
                    score(
                        "mentally_demanding"
                    ),

                "confident":
                    score(
                        "confident"
                    )
            }

        except ValueError:

            return render(
                request,
                "project4/questionnaire.html",
                {
                    "condition":
                        condition,

                    "error":
                        "Please answer all three questions."
                }
            )

        questionnaires = dict(
            study.questionnaire_responses
        )

        questionnaires[key] = (
            answers
        )

        study.questionnaire_responses = (
            questionnaires
        )

        now = time.time()

        if condition == "pairwise":

            started = (
                request.session.get(
                    "project4_pairwise_started"
                )
            )

            if started:

                study.pairwise_time_seconds = (
                    max(
                        0.0,
                        now - float(started)
                    )
                )

        else:

            started = (
                request.session.get(
                    "project4_ranking_started"
                )
            )

            if started:

                study.ranking_time_seconds = (
                    max(
                        0.0,
                        now - float(started)
                    )
                )

        study.save()

        next_condition = (
            _next_after_condition(
                study,
                condition
            )
        )

        return redirect(
            f"project4:{next_condition}"
        )

    return render(
        request,
        "project4/questionnaire.html",
        {
            "condition":
                condition
        }
    )


@require_http_methods(
    ["GET", "POST"]
)
def evaluation(request):

    study = _get_study(
        request
    )

    if not study:

        return redirect(
            "project4:study_intro"
        )

    if (
        len(
            study.pairwise_responses
        )
        < PAIRWISE_ROUNDS
    ):

        return redirect(
            "project4:pairwise"
        )

    if (
        len(
            study.ranking_responses
        )
        < RANKING_ROUNDS
    ):

        return redirect(
            "project4:ranking"
        )

    round_index = len(
        study.evaluation_responses
    )

    if (
        round_index
        >= EVALUATION_ROUNDS
    ):

        return redirect(
            "project4:complete"
        )

    left_id, right_id = (
        study.evaluation_pairs[
            round_index
        ]
    )

    if request.method == "POST":

        try:

            chosen = int(
                request.POST["chosen"]
            )

        except (
            KeyError,
            ValueError
        ):

            raise Http404(
                "Invalid response"
            )

        if chosen not in (
            int(left_id),
            int(right_id)
        ):

            raise Http404(
                "Movie was not presented."
            )

        responses = list(
            study.evaluation_responses
        )

        responses.append(
            {
                "round":
                    round_index + 1,

                "left":
                    int(left_id),

                "right":
                    int(right_id),

                "chosen":
                    chosen
            }
        )

        study.evaluation_responses = (
            responses
        )

        study.save()

        return redirect(
            "project4:evaluation"
        )

    return render(
        request,
        "project4/evaluation.html",
        {
            "left":
                movie_as_dict(
                    left_id
                ),

            "right":
                movie_as_dict(
                    right_id
                ),

            "round_number":
                round_index + 1,

            "total_rounds":
                EVALUATION_ROUNDS,

            "progress":
                round(
                    (
                        round_index
                        / EVALUATION_ROUNDS
                    )
                    * 100
                )
        }
    )


def complete(request):

    study = _get_study(
        request
    )

    if not study:

        return redirect(
            "project4:study_intro"
        )

    if (
        len(
            study.evaluation_responses
        )
        < EVALUATION_ROUNDS
    ):

        return redirect(
            "project4:evaluation"
        )

    if not study.completed:

        (
            _,
            X,
            _,
            _
        ) = get_movie_data()

        pairwise_w = (
            fit_bradley_terry(
                study.pairwise_responses,
                X
            )
        )

        ranking_w = (
            fit_plackett_luce(
                study.ranking_responses,
                X
            )
        )

        pairwise_metrics = (
            evaluate_on_pairwise_choices(
                pairwise_w,
                study.evaluation_responses,
                X
            )
        )

        ranking_metrics = (
            evaluate_on_pairwise_choices(
                ranking_w,
                study.evaluation_responses,
                X
            )
        )

        study.pairwise_weights = (
            pairwise_w.tolist()
        )

        study.ranking_weights = (
            ranking_w.tolist()
        )

        study.pairwise_metrics = (
            pairwise_metrics
        )

        study.ranking_metrics = (
            ranking_metrics
        )

        study.current_stage = (
            "complete"
        )

        study.completed = True

        study.save()

    return render(
        request,
        "project4/complete.html",
        {
            "participant_code":
                study.participant_code,

            "pairwise_metrics":
                study.pairwise_metrics,

            "ranking_metrics":
                study.ranking_metrics,

            "pairwise_time":
                round(
                    study.pairwise_time_seconds,
                    1
                ),

            "ranking_time":
                round(
                    study.ranking_time_seconds,
                    1
                )
        }
    )


def reset_study(request):

    study = _get_study(
        request
    )

    if study:
        study.delete()

    for key in [
        "project4_study_id",
        "project4_pairwise_started",
        "project4_ranking_started"
    ]:

        request.session.pop(
            key,
            None
        )

    return redirect(
        "project4:index"
    )