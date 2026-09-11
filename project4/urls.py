from django.urls import path
from . import views

app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("study/",views.study_intro,name="study_intro"),

    path("pairwise/",views.pairwise,name="pairwise"),

    path("ranking/",views.ranking,name="ranking"),

    path("questionnaire/<str:condition>/",views.questionnaire,name="questionnaire"),

    path("evaluation/",views.evaluation,name="evaluation"),

    path("complete/",views.complete,name="complete"),

    path("reset/",views.reset_study,name="reset_study"),
]
