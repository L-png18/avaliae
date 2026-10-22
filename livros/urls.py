from django.urls import path

from . import views

app_name = "livros"

urlpatterns = [
    path("", views.inicio, name="inicio"),
    path("base-de-dados/", views.base_dados, name="base_dados"),
]
