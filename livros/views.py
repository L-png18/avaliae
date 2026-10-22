from django.conf import settings
from django.shortcuts import render

from .services.dataset import carregar_dataset, resumo_dataset

# Etapas do projeto exibidas na página inicial.
ETAPAS = [
    {
        "numero": 1,
        "titulo": "Base do projeto",
        "descricao": "Site em Django, layout em HTML/CSS e leitura do CSV de livros.",
        "status": "feito",
        "status_texto": "Concluída",
    },
    {
        "numero": 2,
        "titulo": "Pré-processamento de textos",
        "descricao": "Limpeza, minúsculas, tokenização, remoção de stopwords e lematização.",
        "status": "proxima",
        "status_texto": "Próxima",
    },
    {
        "numero": 3,
        "titulo": "Análise de sentimento",
        "descricao": "Classificar comentários como positivos ou negativos com o VADER.",
        "status": "pendente",
        "status_texto": "Pendente",
    },
    {
        "numero": 4,
        "titulo": "Machine Learning",
        "descricao": "Treinar um classificador com TF-IDF e testar novos comentários.",
        "status": "pendente",
        "status_texto": "Pendente",
    },
]


def inicio(request):
    return render(request, "livros/inicio.html", {"etapas": ETAPAS})


def base_dados(request):
    df = carregar_dataset(settings.DATA_PATH)
    return render(request, "livros/base_dados.html", resumo_dataset(df))
