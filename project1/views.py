import traceback

from .ml_engine import run_multiple_models
from django.shortcuts import render
from django.http import JsonResponse
from io import StringIO
import json
import pandas as pd
from django.contrib import messages
from django.shortcuts import redirect
from django.views.decorators.http import require_POST
from .forms import CSVUploadForm


def clean_dataframe(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    unnamed_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    df = df.dropna()
    return df


def load_dataframe_from_session(request):
    df_json = request.session.get("project1_csv")
    if not df_json:
        return None
    return pd.read_json(StringIO(df_json), orient="split")


def detect_task_type(df, target_col):
    y = df[target_col]
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 20:
            return "classification"
        return "regression"
    return "classification"


def build_scatter_datasets(df, x_col, y_col, target_col, task_type):
    datasets = []
    if task_type == "classification":
        for class_name in df[target_col].astype(str).unique().tolist():
            subset = df[df[target_col].astype(str) == class_name]
            datasets.append({
                "label": class_name,
                "data": [
                    {"x": float(row[x_col]), "y": float(row[y_col])}
                    for _, row in subset.iterrows()
                ]
            })
    else:
        datasets.append({
            "label": f"{x_col} vs {y_col}",
            "data": [
                {"x": float(row[x_col]), "y": float(row[y_col])}
                for _, row in df.iterrows()
            ]
        })
    return datasets


def _no_cache(response):
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response


def index(request):
    upload_form = CSVUploadForm()
    context = {
        "upload_form": upload_form,
        "has_data": False,
    }

    df = load_dataframe_from_session(request)

    if df is not None:
        target_col = df.columns[-1]
        feature_cols = list(df.columns[:-1])
        context.update({
            "has_data": True,
            "columns": df.columns.tolist(),
            "rows": df.head(10).values.tolist(),
            "row_count": len(df),
            "col_count": len(df.columns),
            "target_col": target_col,
            "feature_cols": feature_cols,
        })
        task_type = detect_task_type(df, target_col)
        context["detected_task_type"] = task_type
        numeric_feature_cols = [
            col for col in feature_cols
            if pd.api.types.is_numeric_dtype(df[col])
        ]
        if len(numeric_feature_cols) >= 2:
            x_col = numeric_feature_cols[0]
            y_col = numeric_feature_cols[1]
            scatter_datasets = build_scatter_datasets(
                df=df, x_col=x_col, y_col=y_col,
                target_col=target_col, task_type=task_type,
            )
            context["chart_x"] = x_col
            context["chart_y"] = y_col
            context["scatter_datasets_json"] = json.dumps(scatter_datasets)

    return _no_cache(render(request, "project1/index.html", context))


@require_POST
def upload_csv(request):
    """AJAX upload — no navigation, no history entry created."""
    form = CSVUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({"ok": False, "error": "Invalid form submission."}, status=400)

    csv_file = form.cleaned_data["csv_file"]
    try:
        for key in ["project1_csv", "project1_results", "project1_model"]:
            request.session.pop(key, None)

        df = pd.read_csv(csv_file)
        df = clean_dataframe(df)

        if df.shape[1] < 2:
            raise ValueError("CSV must contain at least one feature column and one target column.")

        request.session["project1_csv"] = df.to_json(orient="split")
        request.session.modified = True

        target_col = df.columns[-1]
        feature_cols = list(df.columns[:-1])
        task_type = detect_task_type(df, target_col)
        numeric_feature_cols = [
            col for col in feature_cols
            if pd.api.types.is_numeric_dtype(df[col])
        ]
        scatter_datasets = chart_x = chart_y = None
        if len(numeric_feature_cols) >= 2:
            chart_x = numeric_feature_cols[0]
            chart_y = numeric_feature_cols[1]
            scatter_datasets = build_scatter_datasets(
                df=df, x_col=chart_x, y_col=chart_y,
                target_col=target_col, task_type=task_type,
            )

        return JsonResponse({
            "ok": True,
            "row_count": len(df),
            "col_count": len(df.columns),
            "columns": df.columns.tolist(),
            "rows": df.head(10).values.tolist(),
            "target_col": target_col,
            "feature_cols": feature_cols,
            "detected_task_type": task_type,
            "chart_x": chart_x,
            "chart_y": chart_y,
            "scatter_datasets": scatter_datasets,
        })
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=400)


@require_POST
def reset_session(request):
    for key in ["project1_csv", "project1_results", "project1_model"]:
        request.session.pop(key, None)
    request.session.modified = True
    return JsonResponse({"ok": True})

@require_POST
def run_model(request):
    try:
        df = load_dataframe_from_session(request)

        if df is None:
            return JsonResponse({"ok": False, "error": "No dataset found"})
        body = json.loads(request.body)
        selected_models = body.get("models", [])
        hyperparameters = body.get("hyperparameters", {})
        test_size = float( body.get("test_size", 0.2))
        if not selected_models:
            return JsonResponse({"ok": False,"error": "No models selected"})
        target_col = df.columns[-1]
        task_type = detect_task_type(df,target_col)
        results = run_multiple_models(
            df=df,
            target_col=target_col,
            task_type=task_type,
            selected_models=selected_models,
            test_size=test_size,
            hyperparameters=hyperparameters
        )
        return JsonResponse({ "ok": True,"results": results,"task_type": task_type })
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"ok": False,"error": str(e) })