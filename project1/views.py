from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST

import pandas as pd

from .forms import CSVUploadForm
from .ml_engine import run_ml_pipeline


# ================= HOME =================

def index(request):

    form = CSVUploadForm()

    return render(request, "project1/index.html", {
        "upload_form": form
    })


# ================= LOAD DF =================

def load_df(request):

    data = request.session.get("project1_csv")

    if not data:
        return None

    return pd.read_json(data)


# ================= UPLOAD CSV =================

@require_POST
def upload_csv(request):

    try:

        csv_file = request.FILES["csv_file"]

        df = pd.read_csv(csv_file)

        # save in session
        request.session["project1_csv"] = df.to_json()

        columns = df.columns.tolist()

        rows = df.head(10).values.tolist()

        # ================= SCATTER =================

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        scatter_data = []

        chart_x = ""
        chart_y = ""

        if len(numeric_cols) >= 2:

            chart_x = numeric_cols[0]
            chart_y = numeric_cols[1]

            for i in range(len(df)):

                scatter_data.append({

                    "x": float(df[chart_x].iloc[i]),

                    "y": float(df[chart_y].iloc[i])

                })

        return JsonResponse({

            "ok": True,

            "columns": columns,

            "rows": rows,

            "scatter_data": scatter_data,

            "chart_x": chart_x,

            "chart_y": chart_y

        })

    except Exception as e:

        return JsonResponse({
            "ok": False,
            "error": str(e)
        })


# ================= RUN MODEL =================

@require_POST
def run_model(request):

    try:

        df = load_df(request)

        if df is None:

            return JsonResponse({
                "ok": False,
                "error": "No dataset uploaded"
            })

        target_col = df.columns[-1]

        model_name = request.POST.get("model")

        result = run_ml_pipeline(
            df,
            target_col,
            model_name
        )

        return JsonResponse({

            "ok": True,

            "result": result

        })

    except Exception as e:

        return JsonResponse({
            "ok": False,
            "error": str(e)
        })


# ================= RESET =================

@require_POST
def reset_session(request):

    request.session.flush()

    return JsonResponse({
        "ok": True
    })