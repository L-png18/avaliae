"""Pré-processamento de textos (etapa 2 do projeto).

Segue o slide "Identificar se comentários são positivos ou negativos":
remover pontuação e caracteres especiais, converter para minúsculas, tokenizar,
remover stopwords e lematizar.

Este módulo não depende do Django. Os recursos do NLTK (stopwords, WordNet)
são carregados só quando necessários e podem ser substituídos por funções
próprias nos parâmetros ``stopwords``, ``lematizar`` e ``radicalizar`` (é o que
os testes fazem).

Os textos do dataset estão em inglês, então usamos stopwords e lematizador
em inglês.
"""
import html
import re
from collections import Counter
from functools import lru_cache
from typing import Callable, Iterable, Optional

from .dataset import formatar_inteiro, formatar_percentual

MSG_INSTALAR_NLTK = (
    "A biblioteca NLTK não está instalada. Rode no terminal: "
    "pip install -r requirements.txt"
)
MSG_BAIXAR_NLTK = (
    "Não consegui baixar os recursos do NLTK automaticamente (verifique a "
    "internet). Rode no terminal: python -m nltk.downloader stopwords wordnet omw-1.4"
)

# Palavras de negação. Na análise de sentimento elas importam: "not good" é
# o oposto de "good". Por padrão, não são removidas como stopwords.
NEGACOES = frozenset(
    {
        "no", "nor", "not", "ain", "aren", "aren't", "couldn", "couldn't",
        "didn", "didn't", "doesn", "doesn't", "don", "don't", "hadn", "hadn't",
        "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "mightn",
        "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
        "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
        "won't", "wouldn", "wouldn't",
    }
)

_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_NAO_LETRA = re.compile(r"[^a-z']+")


class RecursosNLTKIndisponiveis(RuntimeError):
    """O NLTK (ou algum de seus recursos) não pôde ser usado."""


# --------------------------------------------------------------------------
# Passos básicos (não precisam de NLTK)
# --------------------------------------------------------------------------
def limpar_texto(texto: str) -> str:
    """Remove HTML, links, números e pontuação; converte para minúsculas.

    O apóstrofo é mantido para preservar contrações como "don't".
    """
    texto = html.unescape(str(texto))
    texto = _RE_HTML.sub(" ", texto)
    texto = _RE_URL.sub(" ", texto)
    texto = _RE_NAO_LETRA.sub(" ", texto.lower())
    return " ".join(texto.split())


def meu_tokenizador(texto: str, delimitador: Optional[str] = None) -> list[str]:
    """Função 1 do exercício do slide: divide o texto pelo delimitador.

    Com ``delimitador=None`` divide em qualquer espaço em branco.
    """
    return [token for token in texto.split(delimitador) if token]


def _normalizar_token(token: str) -> str:
    token = token.strip("'")
    if token.endswith("'s"):  # possessivo: "book's" -> "book"
        token = token[:-2]
    return token.strip("'")


def tokenizar(texto_limpo: str) -> list[str]:
    """Separa o texto limpo em palavras (tokens)."""
    tokens = (_normalizar_token(t) for t in meu_tokenizador(texto_limpo, " "))
    return [t for t in tokens if t]


def remover_stopwords(lista_tokens: Iterable[str], stopwords: Iterable[str]) -> list[str]:
    """Função 2 do exercício do slide: mantém só os tokens que não são stopwords."""
    conjunto = {palavra.lower() for palavra in stopwords}
    return [token for token in lista_tokens if token.lower() not in conjunto]


def ajustar_stopwords(palavras: Iterable[str], manter_negacoes: bool = True) -> frozenset:
    """Devolve o conjunto de stopwords, sem as negações quando pedido."""
    conjunto = frozenset(palavras)
    return conjunto - NEGACOES if manter_negacoes else conjunto


# --------------------------------------------------------------------------
# Recursos do NLTK (carregados sob demanda)
# --------------------------------------------------------------------------
def _baixar(*recursos: str) -> None:
    try:
        import nltk

        for recurso in recursos:
            nltk.download(recurso, quiet=True)
    except Exception:  # sem internet, proxy etc.: o chamador trata o LookupError
        pass


@lru_cache(maxsize=2)
def obter_stopwords(manter_negacoes: bool = True) -> frozenset:
    """Stopwords do inglês (NLTK), baixando o recurso se for preciso."""
    try:
        from nltk.corpus import stopwords
    except ImportError as erro:
        raise RecursosNLTKIndisponiveis(MSG_INSTALAR_NLTK) from erro

    try:
        palavras = stopwords.words("english")
    except LookupError:
        _baixar("stopwords")
        try:
            palavras = stopwords.words("english")
        except LookupError as erro:
            raise RecursosNLTKIndisponiveis(MSG_BAIXAR_NLTK) from erro
    return ajustar_stopwords(palavras, manter_negacoes)


@lru_cache(maxsize=1)
def obter_lematizador() -> Callable[[str], str]:
    """Lematizador do inglês (WordNet).

    Sem saber a classe gramatical da palavra, tentamos como verbo, depois como
    substantivo e por fim como adjetivo ("was" -> "be", "books" -> "book",
    "better" -> "good"). É uma aproximação simples.
    """
    try:
        from nltk.stem import WordNetLemmatizer
    except ImportError as erro:
        raise RecursosNLTKIndisponiveis(MSG_INSTALAR_NLTK) from erro

    wordnet = WordNetLemmatizer()
    try:
        wordnet.lemmatize("tests", "n")
    except LookupError:
        _baixar("wordnet", "omw-1.4")
        try:
            wordnet.lemmatize("tests", "n")
        except LookupError as erro:
            raise RecursosNLTKIndisponiveis(MSG_BAIXAR_NLTK) from erro

    @lru_cache(maxsize=None)
    def lematizar(palavra: str) -> str:
        for classe in ("v", "n", "a"):
            lema = wordnet.lemmatize(palavra, classe)
            if lema != palavra:
                return lema
        return palavra

    return lematizar


@lru_cache(maxsize=1)
def obter_radicalizador() -> Callable[[str], str]:
    """Stemmer de Porter (inglês). Não precisa baixar nada."""
    try:
        from nltk.stem import PorterStemmer
    except ImportError as erro:
        raise RecursosNLTKIndisponiveis(MSG_INSTALAR_NLTK) from erro
    return lru_cache(maxsize=None)(PorterStemmer().stem)


# --------------------------------------------------------------------------
# Pipeline completo
# --------------------------------------------------------------------------
def executar_pipeline(
    texto: str,
    *,
    manter_negacoes: bool = True,
    stopwords: Optional[Iterable[str]] = None,
    lematizar: Optional[Callable[[str], str]] = None,
    radicalizar: Optional[Callable[[str], str]] = None,
) -> dict:
    """Aplica todos os passos a um texto e devolve o resultado de cada um."""
    if stopwords is None:
        stopwords = obter_stopwords(manter_negacoes)
    lematizar = lematizar or obter_lematizador()
    radicalizar = radicalizar or obter_radicalizador()

    conjunto = {palavra.lower() for palavra in stopwords}
    limpo = limpar_texto(texto)
    tokens = tokenizar(limpo)
    sem_stopwords = remover_stopwords(tokens, conjunto)
    removidas = [token for token in tokens if token.lower() in conjunto]

    pares = []
    for palavra in sem_stopwords:
        lema = lematizar(palavra)
        pares.append(
            {
                "palavra": palavra,
                "lema": lema,
                "radical": radicalizar(palavra),
                "mudou": lema != palavra,
            }
        )

    return {
        "original": str(texto),
        "limpo": limpo,
        "tokens": tokens,
        "sem_stopwords": sem_stopwords,
        "removidas": removidas,
        "pares": pares,
        "lematizados": [par["lema"] for par in pares],
        "radicais": [par["radical"] for par in pares],
    }


# --------------------------------------------------------------------------
# Efeito no corpus inteiro
# --------------------------------------------------------------------------
def estatisticas_corpus(
    textos: Iterable[str],
    *,
    manter_negacoes: bool = True,
    stopwords: Optional[Iterable[str]] = None,
    lematizar: Optional[Callable[[str], str]] = None,
    radicalizar: Optional[Callable[[str], str]] = None,
    top: int = 15,
) -> dict:
    """Compara tamanho do texto e do vocabulário a cada etapa do pré-processamento."""
    if stopwords is None:
        stopwords = obter_stopwords(manter_negacoes)
    lematizar = lematizar or obter_lematizador()
    radicalizar = radicalizar or obter_radicalizador()
    conjunto = {palavra.lower() for palavra in stopwords}

    chaves = ("tokens", "sem_stopwords", "lemas", "radicais")
    contadores = {chave: Counter() for chave in chaves}
    totais = dict.fromkeys(chaves, 0)
    total_textos = 0

    for texto in textos:
        total_textos += 1
        tokens = tokenizar(limpar_texto(texto))
        sem_stopwords = remover_stopwords(tokens, conjunto)
        listas = {
            "tokens": tokens,
            "sem_stopwords": sem_stopwords,
            "lemas": [lematizar(t) for t in sem_stopwords],
            "radicais": [radicalizar(t) for t in sem_stopwords],
        }
        for chave, lista in listas.items():
            contadores[chave].update(lista)
            totais[chave] += len(lista)

    nomes = {
        "tokens": "Tokenização (após a limpeza)",
        "sem_stopwords": "Sem stopwords",
        "lemas": "Lematização",
        "radicais": "Stemming (comparação)",
    }
    vocabulario_base = len(contadores["tokens"])
    etapas = []
    for chave in chaves:
        vocabulario = len(contadores[chave])
        if vocabulario_base:
            largura = round(vocabulario / vocabulario_base * 100)
            reducao = (1 - vocabulario / vocabulario_base) * 100
        else:
            largura, reducao = 0, 0.0
        etapas.append(
            {
                "nome": nomes[chave],
                "total_tokens": formatar_inteiro(totais[chave]),
                "vocabulario": formatar_inteiro(vocabulario),
                "reducao": "—" if chave == "tokens" else formatar_percentual(reducao),
                "largura": largura,
            }
        )

    return {
        "total_textos": formatar_inteiro(total_textos),
        "etapas": etapas,
        "mais_frequentes": [
            {"palavra": palavra, "frequencia": formatar_inteiro(freq)}
            for palavra, freq in contadores["lemas"].most_common(top)
        ],
    }


def resumo_corpus(df, manter_negacoes: bool = True, tamanho_amostra: int = 3000, exemplos: int = 5) -> dict:
    """Roda o pré-processamento numa amostra fixa de avaliações do dataset."""
    amostra = df.sample(n=min(tamanho_amostra, len(df)), random_state=42)
    textos = amostra["review/text"].fillna("").tolist()

    stopwords = obter_stopwords(manter_negacoes)
    lematizar = obter_lematizador()
    radicalizar = obter_radicalizador()

    resumo = estatisticas_corpus(
        textos, stopwords=stopwords, lematizar=lematizar, radicalizar=radicalizar
    )
    curtos = [t for t in textos if 80 <= len(t) <= 300][:exemplos]
    resumo["exemplos"] = [
        {
            "original": t,
            "final": " ".join(
                executar_pipeline(
                    t, stopwords=stopwords, lematizar=lematizar, radicalizar=radicalizar
                )["lematizados"]
            ),
        }
        for t in curtos
    ]
    return resumo
