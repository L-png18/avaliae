from functools import lru_cache

from django.conf import settings
from django.shortcuts import render

from .services.classificador import (
    ScikitLearnIndisponivel,
    preparar_dataset,
    prever,
    sortear_avaliacao_de_teste,
    treinar,
)
from .services.dataset import (
    carregar_dataset,
    resumo_dataset,
    sortear_avaliacao,
    sortear_avaliacao_com_nota,
)
from .services.preprocessamento import (
    RecursosNLTKIndisponiveis,
    executar_pipeline,
    resumo_corpus,
)
from .services.sentimento import (
    analisar_texto,
    classificar_nota,
    resumo_sentimento_corpus,
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
        "status": "feito",
        "status_texto": "Concluída",
    },
    {
        "numero": 3,
        "titulo": "Análise de sentimento",
        "descricao": "Classificar comentários como positivos ou negativos com o VADER.",
        "status": "feito",
        "status_texto": "Concluída",
    },
    {
        "numero": 4,
        "titulo": "Machine Learning",
        "descricao": "Treinar um classificador com TF-IDF e testar novos comentários.",
        "status": "feito",
        "status_texto": "Concluída",
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


@lru_cache(maxsize=1)
def _validacao_sentimento_em_cache(caminho):
    """Roda o VADER em 3.000 avaliações (alguns segundos); guarda para as próximas visitas."""
    return resumo_sentimento_corpus(carregar_dataset(caminho))


def sentimento(request):
    texto = request.GET.get("texto", "").strip()[:TAMANHO_MAXIMO_TEXTO]
    nota_csv = None

    if request.GET.get("sortear"):
        texto, nota = sortear_avaliacao_com_nota(carregar_dataset(settings.DATA_PATH))
        classe_nota = classificar_nota(nota)
        if classe_nota:
            nota_csv = {"nota": f"{float(nota):.0f}", "classe": classe_nota}
    elif not texto:
        texto = TEXTO_EXEMPLO

    contexto = {"texto": texto, "nota_csv": nota_csv}
    try:
        resultado = analisar_texto(texto)
        if nota_csv:
            nota_csv["concorda"] = nota_csv["classe"] == resultado["original"]["classe"]
        contexto["resultado"] = resultado
    except RecursosNLTKIndisponiveis as erro:
        contexto["erro_nltk"] = str(erro)
    return render(request, "livros/sentimento.html", contexto)


def sentimento_corpus(request):
    contexto = {}
    try:
        contexto.update(_validacao_sentimento_em_cache(settings.DATA_PATH))
    except RecursosNLTKIndisponiveis as erro:
        contexto["erro_nltk"] = str(erro)
    return render(request, "livros/sentimento_corpus.html", contexto)


@lru_cache(maxsize=1)
def _treinamento_em_cache(caminho):
    """Treina Naive Bayes e SVM (cerca de 15 s); guarda para as próximas visitas."""
    dados, descartes = preparar_dataset(carregar_dataset(caminho))
    return treinar(dados, descartes=descartes)


def machine_learning(request):
    contexto = {}
    try:
        contexto.update(_treinamento_em_cache(settings.DATA_PATH).resumo)
    except (RecursosNLTKIndisponiveis, ScikitLearnIndisponivel) as erro:
        contexto["erro_nltk"] = str(erro)
    return render(request, "livros/machine_learning.html", contexto)


def ml_testar(request):
    texto = request.GET.get("texto", "").strip()[:TAMANHO_MAXIMO_TEXTO]
    sortear = request.GET.get("sortear")
    contexto = {"texto": texto, "classe_verdadeira": None}
    try:
        treinamento = _treinamento_em_cache(settings.DATA_PATH)
        if sortear in ("negativo", "positivo"):
            # sorteia só do conjunto de teste: o modelo nunca viu esses textos
            texto, classe = sortear_avaliacao_de_teste(treinamento, sortear)
            contexto.update({"texto": texto, "classe_verdadeira": classe})
        elif not texto:
            texto = TEXTO_EXEMPLO
            contexto["texto"] = texto
        resultado = prever(treinamento, texto)
        if contexto["classe_verdadeira"] and "svm" in resultado:
            verdadeira = contexto["classe_verdadeira"]
            resultado["acertos"] = {
                "naive_bayes": resultado["naive_bayes"]["classe"] == verdadeira,
                "svm": resultado["svm"]["classe"] == verdadeira,
                "vader": resultado["vader"]["binaria"] == verdadeira,
            }
        contexto["resultado"] = resultado
    except (RecursosNLTKIndisponiveis, ScikitLearnIndisponivel) as erro:
        contexto["erro_nltk"] = str(erro)
    return render(request, "livros/ml_testar.html", contexto)
