from django.urls import path

from . import views

app_name = "livros"

urlpatterns = [
    path("", views.inicio, name="inicio"),
    path("base-de-dados/", views.base_dados, name="base_dados"),
    path("pre-processamento/", views.pre_processamento, name="pre_processamento"),
    path("pre-processamento/corpus/", views.corpus, name="corpus"),
    path("sentimento/", views.sentimento, name="sentimento"),
    path("sentimento/validacao/", views.sentimento_corpus, name="sentimento_corpus"),
    path("machine-learning/", views.machine_learning, name="machine_learning"),
    path("machine-learning/testar/", views.ml_testar, name="ml_testar"),
]
