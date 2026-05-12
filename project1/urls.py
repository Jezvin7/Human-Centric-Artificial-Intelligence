from django.urls import path
from .views import run_model
from . import views

app_name = 'project1'

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload_csv, name="upload_csv"),
    path("reset/", views.reset_session, name="reset_session"),
    path("run_model/", views.run_model, name="run_model"),
]