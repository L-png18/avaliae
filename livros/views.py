from functools import lru_cache

from django.conf import settings
from django.shortcuts import render

from .services.dataset import carregar_dataset, resumo_dataset, sortear_avaliacao
from .services.preprocessamento import (
    RecursosNLTKIndisponiveis,
    executar_pipeline,
    resumo_corpus,
)

TEXTO_EXEMPLO = (
    "I bought this book last week and couldn't stop reading it! The characters "
    "were amazing, but the ending wasn't what I was expecting :( 10/10 would "
    "recommend."
)
TAMANHO_MAXIMO_TEXTO = 5000

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
        "status_texto": "Em andamento",
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


def _manter_negacoes(request) -> bool:
    """A caixa 'manter negações' vem marcada por padrão (antes de enviar o formulário)."""
    if "enviado" in request.GET:
        return request.GET.get("negacoes") == "on"
    return True


@lru_cache(maxsize=2)
def _resumo_corpus_em_cache(caminho, manter_negacoes):
    """O cálculo leva alguns segundos; guardamos o resultado para as próximas visitas."""
    return resumo_corpus(carregar_dataset(caminho), manter_negacoes=manter_negacoes)


def inicio(request):
    return render(request, "livros/inicio.html", {"etapas": ETAPAS})


def base_dados(request):
    df = carregar_dataset(settings.DATA_PATH)
    return render(request, "livros/base_dados.html", resumo_dataset(df))


def pre_processamento(request):
    manter_negacoes = _manter_negacoes(request)
    texto = request.GET.get("texto", "").strip()[:TAMANHO_MAXIMO_TEXTO]
    sorteado = False

    if request.GET.get("sortear"):
        texto = sortear_avaliacao(carregar_dataset(settings.DATA_PATH))
        sorteado = True
    elif not texto:
        texto = TEXTO_EXEMPLO

    contexto = {
        "texto": texto,
        "manter_negacoes": manter_negacoes,
        "sorteado": sorteado,
    }
    try:
        contexto["resultado"] = executar_pipeline(texto, manter_negacoes=manter_negacoes)
    except RecursosNLTKIndisponiveis as erro:
        contexto["erro_nltk"] = str(erro)
    return render(request, "livros/pre_processamento.html", contexto)


def corpus(request):
    manter_negacoes = _manter_negacoes(request)
    contexto = {"manter_negacoes": manter_negacoes}
    try:
        contexto.update(_resumo_corpus_em_cache(settings.DATA_PATH, manter_negacoes))
    except RecursosNLTKIndisponiveis as erro:
        contexto["erro_nltk"] = str(erro)
    return render(request, "livros/corpus.html", contexto)
