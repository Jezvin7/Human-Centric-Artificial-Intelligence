from pathlib import Path
import json
import subprocess
import sys
from datetime import datetime
from django.shortcuts import render,  redirect


def index(request):
    project_dir = Path(__file__).resolve().parent
    saved_models_dir = project_dir / "ml" / "saved_models"
    baseline_results_path = (saved_models_dir / "baseline_results.json")
    expert_results_path = (saved_models_dir / "expert_results.json")
    l2d_results_path = (saved_models_dir / "l2d_results.json")
    active_results_path = (saved_models_dir/ "active_learning_results.json")
    human_results_path = (saved_models_dir/ "human_expert_results.json")


    # =========================================================
    # TASK 1 VARIABLES
    # =========================================================

    baseline_results = None
    baseline_class_metrics = []
    baseline_confusion_rows = []


    # =========================================================
    # TASK 2 VARIABLES
    # =========================================================

    expert_results = None
    expert_class_metrics = []
    expert_confusion_rows = []
    expert_profile = []
    comparison_rows = []


    # =========================================================
    # TASK 3 VARIABLES
    # =========================================================

    l2d_results = None
    l2d_class_results = []

    team_improvement = None
    ai_covered_accuracy_percent = None
    expert_deferred_accuracy_percent = None

    useful_deferral_rate_percent = None
    harmful_deferral_rate_percent = None

    useful_harmful_ratio = None
    oracle_gap = None


    # =========================================================
    # TASK 4 VARIABLES
    # =========================================================

    active_results = None
    active_learning_rows = []

    best_active_point = None
    final_active_point = None

    best_active_advantage = None
    best_gap_to_full_l2d = None
    best_query_fraction = None
    labels_saved_percent = None

    # =========================================================
    # TASK 5 VARIABLES
    # =========================================================

    human_results = None

    # =========================================================
    # TASK 1 — BASELINE RESULTS
    # =========================================================

    if baseline_results_path.exists():

        with open(
            baseline_results_path,
            "r",
            encoding="utf-8",
        ) as file:

            baseline_results = json.load(file)


        for class_name in baseline_results["class_names"]:

            metrics = (
                baseline_results
                ["classification_report"]
                [class_name]
            )

            baseline_class_metrics.append(
                {
                    "name": class_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1-score"],
                    "support": int(metrics["support"]),
                }
            )


        baseline_confusion_rows = list(
            zip(
                baseline_results["class_names"],
                baseline_results["confusion_matrix"],
            )
        )


    # =========================================================
    # TASK 2 — EXPERT RESULTS
    # =========================================================

    if expert_results_path.exists():

        with open(
            expert_results_path,
            "r",
            encoding="utf-8",
        ) as file:

            expert_results = json.load(file)


        for class_name in expert_results["class_names"]:

            metrics = (
                expert_results
                ["classification_report"]
                [class_name]
            )

            expert_class_metrics.append(
                {
                    "name": class_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1-score"],
                    "support": int(metrics["support"]),
                }
            )


        expert_confusion_rows = list(
            zip(
                expert_results["class_names"],
                expert_results["confusion_matrix"],
            )
        )


        for class_name in expert_results["class_names"]:

            competence = (
                expert_results
                ["competence_profile"]
                [class_name]
            )

            expert_profile.append(
                {
                    "name": class_name,
                    "accuracy_percent": competence * 100,
                }
            )


    # =========================================================
    # TASK 1 VS TASK 2
    # =========================================================

    if baseline_results and expert_results:

        for class_name in baseline_results["class_names"]:

            ai_recall = (
                baseline_results
                ["classification_report"]
                [class_name]
                ["recall"]
            )

            expert_recall = (
                expert_results
                ["classification_report"]
                [class_name]
                ["recall"]
            )


            if ai_recall > expert_recall:
                better_system = "AI"

            elif expert_recall > ai_recall:
                better_system = "Expert"

            else:
                better_system = "Equal"


            comparison_rows.append(
                {
                    "name": class_name,
                    "ai_recall": ai_recall,
                    "expert_recall": expert_recall,
                    "better": better_system,
                }
            )


    # =========================================================
    # TASK 3 — LEARNING TO DEFER
    # =========================================================

    if l2d_results_path.exists():

        with open(
            l2d_results_path,
            "r",
            encoding="utf-8",
        ) as file:

            l2d_results = json.load(file)


        if l2d_results["ai_covered_accuracy"] is not None:

            ai_covered_accuracy_percent = (
                l2d_results["ai_covered_accuracy"]
                * 100
            )


        if l2d_results["expert_deferred_accuracy"] is not None:

            expert_deferred_accuracy_percent = (
                l2d_results["expert_deferred_accuracy"]
                * 100
            )


        useful_deferral_rate_percent = (
            l2d_results["useful_deferral_rate"]
            * 100
        )


        harmful_deferral_rate_percent = (
            l2d_results["harmful_deferral_rate"]
            * 100
        )


        team_improvement = (
            l2d_results["team_accuracy_percent"]
            -
            l2d_results["ai_accuracy_percent"]
        )


        oracle_gap = (
            l2d_results["oracle_accuracy_percent"]
            -
            l2d_results["team_accuracy_percent"]
        )


        harmful_count = (
            l2d_results["harmful_deferrals"]
        )

        useful_count = (
            l2d_results["useful_deferrals"]
        )


        if harmful_count > 0:

            useful_harmful_ratio = (
                useful_count
                /
                harmful_count
            )


        for item in l2d_results["class_results"]:

            l2d_class_results.append(
                {
                    "class": item["class"],
                    "samples": item["samples"],
                    "deferred": item["deferred"],

                    "deferral_rate_percent":
                        item["deferral_rate"] * 100,

                    "team_accuracy_percent":
                        item["team_accuracy"] * 100,
                }
            )


    # =========================================================
    # TASK 4 — ACTIVE LEARNING
    # =========================================================

    if active_results_path.exists():

        with open(
            active_results_path,
            "r",
            encoding="utf-8",
        ) as file:

            active_results = json.load(file)


        # -----------------------------------------------------
        # Build table rows for all expert-query budgets
        # -----------------------------------------------------

        for experiment in active_results["experiments"]:

            random_result = (
                experiment["random"]
            )

            uncertainty_result = (
                experiment["competence_uncertainty"]
            )

            deferral_result = (
                experiment["deferral_aware_diversity"]
            )


            random_recovery = (
                experiment["random_gain_recovery"]
            )

            uncertainty_recovery = (
                experiment["uncertainty_gain_recovery"]
            )

            deferral_recovery = (
                experiment["deferral_gain_recovery"]
            )


            row = {

                "budget":
                    experiment["query_budget"],

                "query_fraction":
                    experiment["query_fraction_percent"],


                # Random
                "random_accuracy":
                    random_result["team_accuracy_percent"],

                "random_routing":
                    random_result["routing_accuracy_percent"],

                "random_deferral":
                    random_result["deferral_rate_percent"],

                "random_recovery":
                    (
                        random_recovery * 100
                        if random_recovery is not None
                        else None
                    ),


                # Standard active learning
                "uncertainty_accuracy":
                    uncertainty_result[
                        "team_accuracy_percent"
                    ],

                "uncertainty_routing":
                    uncertainty_result[
                        "routing_accuracy_percent"
                    ],

                "uncertainty_deferral":
                    uncertainty_result[
                        "deferral_rate_percent"
                    ],

                "uncertainty_recovery":
                    (
                        uncertainty_recovery * 100
                        if uncertainty_recovery is not None
                        else None
                    ),


                # Deferral-aware experiment
                "deferral_accuracy":
                    deferral_result[
                        "team_accuracy_percent"
                    ],

                "deferral_routing":
                    deferral_result[
                        "routing_accuracy_percent"
                    ],

                "deferral_rate":
                    deferral_result[
                        "deferral_rate_percent"
                    ],

                "deferral_recovery":
                    (
                        deferral_recovery * 100
                        if deferral_recovery is not None
                        else None
                    ),
            }


            # Advantage of uncertainty sampling over random
            row["uncertainty_advantage"] = (
                row["uncertainty_accuracy"]
                -
                row["random_accuracy"]
            )


            active_learning_rows.append(
                row
            )

        if active_learning_rows:

            best_active_point = max(
                active_learning_rows,
                key=lambda row:
                    row["uncertainty_advantage"],
            )


            final_active_point = (
                active_learning_rows[-1]
            )


            best_active_advantage = (
                best_active_point[
                    "uncertainty_advantage"
                ]
            )


            best_query_fraction = (
                best_active_point[
                    "query_fraction"
                ]
            )


            labels_saved_percent = (
                100
                -
                best_query_fraction
            )


            # -------------------------------------------------
            # Difference from full-information Task 3 system
            # -------------------------------------------------

            if (
                active_results[
                    "full_l2d_accuracy_percent"
                ]
                is not None
            ):

                best_gap_to_full_l2d = (
                    active_results[
                        "full_l2d_accuracy_percent"
                    ]
                    -
                    best_active_point[
                        "uncertainty_accuracy"
                    ]
                )
    run_message = request.session.pop(
    "project3_run_message",
    None,
    )

    run_success = request.session.pop(
        "project3_run_success",
        None,
    )

    if human_results_path.exists():
        with open(
            human_results_path,
            "r",
            encoding="utf-8",
        ) as file:
            human_results = json.load(file)

    human_message = request.session.pop(
                    "project3_human_message",
                    None,
                    )

    # =========================================================
    # CONTEXT
    # =========================================================

    context = {
        
        # Task 1
        "baseline_results":
            baseline_results,

        "baseline_class_metrics":
            baseline_class_metrics,

        "baseline_confusion_rows":
            baseline_confusion_rows,


        # Task 2
        "expert_results":
            expert_results,

        "expert_class_metrics":
            expert_class_metrics,

        "expert_confusion_rows":
            expert_confusion_rows,

        "expert_profile":
            expert_profile,

        "comparison_rows":
            comparison_rows,


        # Task 3
        "l2d_results":
            l2d_results,

        "l2d_class_results":
            l2d_class_results,

        "team_improvement":
            team_improvement,

        "ai_covered_accuracy_percent":
            ai_covered_accuracy_percent,

        "expert_deferred_accuracy_percent":
            expert_deferred_accuracy_percent,

        "useful_deferral_rate_percent":
            useful_deferral_rate_percent,

        "harmful_deferral_rate_percent":
            harmful_deferral_rate_percent,

        "useful_harmful_ratio":
            useful_harmful_ratio,

        "oracle_gap":
            oracle_gap,


        # Task 4
        "active_results":
            active_results,

        "active_learning_rows":
            active_learning_rows,

        "best_active_point":
            best_active_point,

        "final_active_point":
            final_active_point,

        "best_active_advantage":
            best_active_advantage,

        "best_gap_to_full_l2d":
            best_gap_to_full_l2d,

        "best_query_fraction":
            best_query_fraction,

        "labels_saved_percent":
            labels_saved_percent,

        "run_message": run_message,
        "run_success": run_success,

        "human_results": human_results,
        "human_message": human_message,
    }


    return render(
        request,
        "project3/index.html",
        context,
    )


def run_all_analysis(request):

    if request.method != "POST":
        return redirect(
            "project3:index"
        )

    project_dir = Path(
        __file__
    ).resolve().parent

    root_dir = (
        project_dir.parent
    )

    run_all_path = (
        project_dir
        / "ml"
        / "run_all.py"
    )

    try:

        result = subprocess.run(
            [
                sys.executable,
                str(run_all_path),
            ],
            cwd=root_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:

            request.session[
                "project3_run_message"
            ] = (
                "Full analysis completed successfully. "
                "Tasks 1–4 and the project report "
                "have been updated."
            )

            request.session[
                "project3_run_success"
            ] = True

        else:

            request.session[
                "project3_run_message"
            ] = (
                "The analysis stopped because one "
                "of the experiment scripts failed."
            )

            request.session[
                "project3_run_success"
            ] = False

            print(
                "Project 3 pipeline error:"
            )

            print(
                result.stdout
            )

            print(
                result.stderr
            )

    except Exception as error:

        request.session[
            "project3_run_message"
        ] = (
            f"Could not run the analysis: {error}"
        )

        request.session[
            "project3_run_success"
        ] = False

    return redirect(
        "project3:index"
    )

# ============================================================
# TASK 5 - HUMAN EXPERT INTERFACE
# ============================================================


# ============================================================
# TASK 5 - SAVE HUMAN EXPERT RESULTS
# ============================================================

def finalize_human_session(
    project_dir,
    query_pool,
    responses,
):

    class_names = [
        "World",
        "Sports",
        "Business",
        "Sci/Tech",
    ]


    total_answers = len(
        responses
    )


    correct_answers = sum(
        1
        for response in responses
        if response["correct"]
    )


    if total_answers > 0:

        human_accuracy = (
            correct_answers
            /
            total_answers
            *
            100
        )

    else:

        human_accuracy = None


    # ========================================================
    # CLASS-WISE HUMAN PERFORMANCE
    # ========================================================

    per_class_results = []


    for class_index, class_name in enumerate(
        class_names
    ):

        class_responses = [
            response
            for response in responses
            if int(
                response["true_label"]
            )
            ==
            class_index
        ]


        class_total = len(
            class_responses
        )


        class_correct = sum(
            1
            for response in class_responses
            if response["correct"]
        )


        if class_total > 0:

            class_accuracy = (
                class_correct
                /
                class_total
                *
                100
            )

        else:

            class_accuracy = None


        per_class_results.append(
            {
                "class":
                    class_name,

                "queries":
                    class_total,

                "correct":
                    class_correct,

                "accuracy_percent":
                    (
                        round(
                            class_accuracy,
                            2,
                        )
                        if class_accuracy
                        is not None
                        else None
                    ),
            }
        )


    # ========================================================
    # RESULT
    # ========================================================

    results = {

        "task":
            (
                "Task 5 - Active Learning "
                "with Human Expert"
            ),

        "human_queries":
            total_answers,

        "correct_labels":
            correct_answers,

        "incorrect_labels":
            (
                total_answers
                -
                correct_answers
            ),

        "human_accuracy_percent":
            (
                round(
                    human_accuracy,
                    2,
                )
                if human_accuracy
                is not None
                else None
            ),

        "query_pool_size":
            len(
                query_pool
            ),

        "acquisition_strategy":
            (
                "Competence uncertainty sampling"
            ),

        "per_class_results":
            per_class_results,

        "completed_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),
    }


    # ========================================================
    # SAVE FILES
    # ========================================================

    saved_models_dir = (
        project_dir
        / "ml"
        / "saved_models"
    )


    results_path = (
        saved_models_dir
        / "human_expert_results.json"
    )


    log_path = (
        saved_models_dir
        / "human_expert_query_log.json"
    )


    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )


    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            responses,
            file,
            indent=4,
        )


    # ========================================================
    # REGENERATE PDF
    # ========================================================

    report_script = (
        project_dir
        / "ml"
        / "generate_report.py"
    )


    try:

        report_result = subprocess.run(
            [
                sys.executable,
                str(report_script),
            ],
            cwd=project_dir.parent,
            capture_output=True,
            text=True,
        )


        if report_result.returncode == 0:

            report_updated = True

        else:

            report_updated = False

            print(
                report_result.stdout
            )

            print(
                report_result.stderr
            )


    except Exception as error:

        report_updated = False

        print(
            "Task 5 report generation error:",
            error,
        )


    return (
        results,
        report_updated,
    )


def human_expert(request):

    project_dir = (
        Path(__file__).resolve().parent
    )

    pool_path = (
        project_dir
        / "ml"
        / "saved_models"
        / "human_query_pool.json"
    )


    # ========================================================
    # QUERY POOL NOT AVAILABLE
    # ========================================================

    if not pool_path.exists():

        return render(
            request,
            "project3/human_expert.html",
            {
                "pool_missing": True,
            },
        )


    with open(
        pool_path,
        "r",
        encoding="utf-8",
    ) as file:

        query_pool = json.load(file)


    session_key = (
        "project3_human_expert"
    )


    state = request.session.get(
        session_key
    )


    if state is None:

        state = {
            "current": 0,
            "responses": [],
            "finished": False,
        }


    # ========================================================
    # POST ACTIONS
    # ========================================================

    if request.method == "POST":

        action = request.POST.get(
            "action"
        )


        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        if action == "reset":

            state = {
                "current": 0,
                "responses": [],
                "finished": False,
            }


            request.session[
                session_key
            ] = state


            request.session.modified = True


            return redirect(
                "project3:human_expert"
            )


        # ----------------------------------------------------
        # HUMAN PROVIDES LABEL
        # ----------------------------------------------------

        if action == "label":

            try:

                human_label = int(
                    request.POST.get(
                        "label"
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                return redirect(
                    "project3:human_expert"
                )


            if human_label not in [
                0,
                1,
                2,
                3,
            ]:

                return redirect(
                    "project3:human_expert"
                )


            current_position = (
                state["current"]
            )


            if (
                current_position
                >=
                len(query_pool)
            ):

                state[
                    "finished"
                ] = True


                request.session[
                    session_key
                ] = state


                return redirect(
                    "project3:human_expert"
                )


            sample = (
                query_pool[
                    current_position
                ]
            )


            true_label = int(
                sample[
                    "true_label"
                ]
            )


            response = {

                "sample_index":
                    int(
                        sample[
                            "index"
                        ]
                    ),

                "human_label":
                    human_label,

                "true_label":
                    true_label,

                "correct":
                    (
                        human_label
                        ==
                        true_label
                    ),
            }


            state[
                "responses"
            ].append(
                response
            )


            state[
                "current"
            ] += 1


            if (
                state["current"]
                >=
                len(query_pool)
            ):

                state[
                    "finished"
                ] = True


                (
                    results,
                    report_updated,
                ) = finalize_human_session(
                    project_dir,
                    query_pool,
                    state["responses"],
                )


                state[
                    "saved_results"
                ] = results


                request.session[
                    session_key
                ] = state


                request.session.modified = True


                if report_updated:

                    request.session[
                        "project3_human_message"
                    ] = (
                        "Human expert experiment completed. "
                        "Task 5 results and the PDF report "
                        "were updated successfully."
                    )

                else:

                    request.session[
                        "project3_human_message"
                    ] = (
                        "Human expert results were saved, "
                        "but the PDF report could not be regenerated."
                    )


                return redirect(
                    "project3:index"
                )


            request.session[
                session_key
            ] = state


            request.session.modified = True


            return redirect(
                "project3:human_expert"
            )


        # ----------------------------------------------------
        # FINISH EARLY
        # ----------------------------------------------------

        if action == "finish":

            if not state[
                "responses"
            ]:

                return redirect(
                    "project3:human_expert"
                )


            state[
                "finished"
            ] = True


            (
                results,
                report_updated,
            ) = finalize_human_session(
                project_dir,
                query_pool,
                state["responses"],
            )


            state[
                "saved_results"
            ] = results


            request.session[
                session_key
            ] = state


            request.session.modified = True


            if report_updated:

                request.session[
                    "project3_human_message"
                ] = (
                    "Human expert experiment completed. "
                    "Task 5 results and the PDF report "
                    "were updated successfully."
                )

            else:

                request.session[
                    "project3_human_message"
                ] = (
                    "Human expert results were saved, "
                    "but the PDF report could not be regenerated."
                )


            return redirect(
                "project3:index"
            )


    # ========================================================
    # SESSION STATISTICS
    # ========================================================

    responses = (
        state["responses"]
    )


    answered = len(
        responses
    )


    correct_answers = sum(
        1
        for response in responses
        if response["correct"]
    )


    human_accuracy = None


    if answered > 0:

        human_accuracy = (
            correct_answers
            /
            answered
            *
            100
        )


    # ========================================================
    # CURRENT ARTICLE
    # ========================================================

    article = None


    if (
        not state[
            "finished"
        ]
        and
        state[
            "current"
        ]
        <
        len(query_pool)
    ):

        article = (
            query_pool[
                state[
                    "current"
                ]
            ]
        )


    context = {

        "pool_missing":
            False,

        "state":
            state,

        "article":
            article,

        "answered":
            answered,

        "correct_answers":
            correct_answers,

        "human_accuracy":
            human_accuracy,

        "total_queries":
            len(query_pool),

        "class_names": [
            "World",
            "Sports",
            "Business",
            "Sci/Tech",
        ],
    }


    return render(
        request,
        "project3/human_expert.html",
        context,
    )