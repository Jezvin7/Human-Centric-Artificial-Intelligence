# Human-Centric-Artificial-Intelligence
Group Members - Jesvin Varghese , Shashank Mallappa Gujjal

# Project Overview
This repository is a centralized Django-based web application designed to host five distinct modules for the Human-Centric Artificial Intelligence course. The portal focuses on the intersection of machine learning efficiency and human-centric design, providing an interactive dashboard to navigate through various AI implementations.

# Project 1: Automated Machine Learning 
**Key Functionalities**
  * CSV Data Ingestion: Seamlessly upload and process scalar datasets.
  * Dataset Overview: Automated statistical summary of features and targets.
  * Data Preview: Interactive table view of the raw dataset structure.
  * Feature Visualization: Dynamic scatter plots and data distribution charts (via Matplotlib/ChartJS).
  * Multi-model training and comparison using scikit-learn.
  * User-controlled train/test split and basic hyperparameter testing.
  * Model evaluation with relevant metrics and performance charts.
  * Validation, session reset, and loading feedback.

# Project 2: Explainability
**Key Functionalities**
* Uses the Palmer Penguins dataset to predict penguin species.
* Implements interpretable model selection using decision trees and logistic regression.
* Provides a λ slider to balance test accuracy and model complexity.
* Displays selected model accuracy, complexity, and selection score.
* Visualizes the selected decision tree with a clean, readable plot.
* Generates counterfactual explanations for selected test examples and target species.
* Provides PDP and ALE feature effect plots for numerical features.

# Project 3: Active Learning for Learning-to-Defer
**Key Functionalities**
* Uses the AG News dataset for four-class news classification.
* Implements a TF-IDF and Logistic Regression baseline classifier.
* Simulates an imperfect expert with complementary class-specific strengths.
* Implements Learning-to-Defer to decide between AI prediction and expert deferral.
* Evaluates team accuracy, coverage, deferral rate, routing accuracy, and useful/harmful deferrals.
* Implements Active Learning with random, competence uncertainty, and deferral-aware query strategies.
* Provides a Run Full Analysis option to rerun Tasks 1–4 and refresh results.
* Automatically generates and provides a downloadable PDF report with design choices, experiments, and results.

# Project 4: Preference Elicitation
**Key Functionalities**
* Uses the IMDB 5000 Movie Dataset for movie preference modeling.
* Represents movies using genres, duration, release year, IMDb score, and content rating.
* Implements a Bradley-Terry model for pairwise movie preferences.
* Extends pairwise preference modeling to full rankings using the Plackett-Luce model.
* Provides two preference elicitation interfaces: pairwise movie selection and ranking sets of 10 movies.
* Uses a counterbalanced within-subject study design to compare both elicitation methods.
* Collects completion time and subjective feedback on ease of use, mental demand, and confidence.
* Evaluates both learned preference models using held-out movie comparisons.
* Provides a complete participant-facing study flow with consent, elicitation tasks, questionnaires, evaluation, and completion page.
* Automatically generates and provides a downloadable PDF report describing the feature representation, preference models, and proposed user study design.
