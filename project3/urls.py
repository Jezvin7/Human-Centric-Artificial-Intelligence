from django.urls import path
from . import views

app_name = "project3"

urlpatterns = [
    path("", views.index, name="index"),
    path("run-all/", views.run_all_analysis,name="run_all_analysis"),
    path("human-expert/",views.human_expert,name="human_expert"),
]