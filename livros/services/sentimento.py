"""Análise de sentimento com o VADER (etapa 3 do projeto).

Segue o slide "Exemplo com VADER (inglês)": o ``SentimentIntensityAnalyzer`` do
NLTK devolve quatro números para cada texto (``neg``, ``neu``, ``pos`` e
``compound``) e a classe sai do ``compound``:

    compound >  0.05  -> positivo
    compound < -0.05  -> negativo
    entre os dois     -> neutro

Atenção: o VADER foi feito para analisar o texto *original*. Ele usa
pontuação ("great!!!"), MAIÚSCULAS, emoticons (":("), negações e
intensificadores ("very") para ajustar a nota, e o pré-processamento da
etapa 2 apaga ou muda parte disso. Por isso este módulo analisa o texto
original e também o pré-processado, para o resultado poder ser comparado.
(No CSV, a acurácia é parecida nas duas versões, mas o resultado muda em
algumas avaliações: pré-processar não é neutro para o VADER.)

Este módulo não depende do Django. O analisador e o pré-processamento podem
ser substituídos pelos parâmetros ``analisador`` e ``preprocessar`` (é o que os
testes fazem).
"""
import html
import re
import string
from functools import lru_cache
from typing import Callable, Iterable, Optional

from .dataset import formatar_inteiro, formatar_percentual
from .preprocessamento import (
    MSG_BAIXAR_NLTK,
    MSG_INSTALAR_NLTK,
    RecursosNLTKIndisponiveis,
    _baixar,
    executar_pipeline,
)

LIMIAR = 0.05  # valor do slide: acima = positivo, abaixo do negativo = negativo

POSITIVO = "positivo"
NEGATIVO = "negativo"
NEUTRO = "neutro"
CLASSES = (NEGATIVO, NEUTRO, POSITIVO)

_RE_CONTRACAO_NEGATIVA = re.compile(r"n't$")
_RE_TAGS = re.compile(r"<[^>]+>")
# o apóstrofo fica: ele faz parte de "don't", "wasn't"…
_PONTUACAO_DAS_BORDAS = string.punctuation.replace("'", "")


# --------------------------------------------------------------------------
# Classes (positivo / negativo / neutro)
# --------------------------------------------------------------------------
def classificar_compound(compound: float, limiar: float = LIMIAR) -> str:
    """Regra do slide: > 0.05 positivo, < -0.05 negativo, senão neutro."""
    if compound > limiar:
        return POSITIVO
    if compound < -limiar:
        return NEGATIVO
    return NEUTRO


def classificar_nota(nota) -> Optional[str]:
    """Transforma a nota do CSV (1 a 5) numa classe, para comparar com o VADER.

    4 e 5 -> positivo, 1 e 2 -> negativo, 3 -> neutro. Nota ausente -> ``None``.
    """
    try:
        nota = float(nota)
    except (TypeError, ValueError):
        return None
    if nota != nota:  # NaN
        return None
    if nota >= 4:
        return POSITIVO
    if nota <= 2:
        return NEGATIVO
    return NEUTRO


def texto_para_exibir(texto: str) -> str:
    """Deixa o texto do CSV legível numa tabela: 'J&oacute;zef<br />' -> 'Józef'.

    Só para mostrar na tela; o VADER recebe o texto exatamente como está no CSV.
    """
    return " ".join(_RE_TAGS.sub(" ", html.unescape(str(texto))).split())


def formatar_decimal(valor: float, casas: int = 3, sinal: bool = False) -> str:
    """0.3641 -> '0,364' (vírgula decimal); com ``sinal=True``: '+0,364'."""
    texto = f"{valor:+.{casas}f}" if sinal else f"{valor:.{casas}f}"
    return texto.replace(".", ",")


# --------------------------------------------------------------------------
# Analisador (carregado sob demanda)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def obter_analisador():
    """``SentimentIntensityAnalyzer`` do NLTK, baixando o léxico se for preciso."""
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
    except ImportError as erro:
        raise RecursosNLTKIndisponiveis(MSG_INSTALAR_NLTK) from erro

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        _baixar("vader_lexicon")
        try:
            return SentimentIntensityAnalyzer()
        except LookupError as erro:
            raise RecursosNLTKIndisponiveis(MSG_BAIXAR_NLTK) from erro


def texto_preprocessado(texto: str) -> str:
    """Texto depois da etapa 2 (limpeza, stopwords e lematização), numa só linha."""
    return " ".join(executar_pipeline(texto)["lematizados"])


# --------------------------------------------------------------------------
# Pontuação de um texto
# --------------------------------------------------------------------------
def pontuar(texto: str, analisador=None) -> dict:
    """Roda o VADER num texto e já devolve a classe e os números formatados."""
    analisador = analisador or obter_analisador()
    p = analisador.polarity_scores(str(texto))
    compound = p["compound"]
    return {
        "compound": compound,
        "compound_texto": formatar_decimal(compound, 4, sinal=True),
        "classe": classificar_compound(compound),
        # posição do marcador no medidor, que vai de -1 (0%) a +1 (100%)
        "posicao": round((compound + 1) / 2 * 100, 1),
        "barras": [
            {"nome": "neg", "rotulo": "Negativo", "valor": formatar_decimal(p["neg"]), "largura": round(p["neg"] * 100)},
            {"nome": "neu", "rotulo": "Neutro", "valor": formatar_decimal(p["neu"]), "largura": round(p["neu"] * 100)},
            {"nome": "pos", "rotulo": "Positivo", "valor": formatar_decimal(p["pos"]), "largura": round(p["pos"] * 100)},
        ],
    }


def destacar_palavras(
    texto: str,
    lexicon: dict,
    negacoes: Iterable[str] = (),
    intensificadores: Iterable[str] = (),
) -> list[dict]:
    """Lista as palavras do texto que o VADER usa para calcular a nota.

    Cada item tem ``palavra``, ``tipo`` (positiva, negativa, negacao ou
    intensificador) e ``valor`` (a valência do léxico, de -4 a +4, quando houver).
    """
    negacoes = set(negacoes)
    intensificadores = set(intensificadores)
    achadas = []
    for pedaco in str(texto).split():
        baixo = pedaco.lower()
        limpa = baixo.strip(_PONTUACAO_DAS_BORDAS)
        # o pedaço inteiro vem primeiro: emoticons como ":(" são só pontuação
        palavra = baixo if baixo in lexicon else (limpa if limpa in lexicon else None)
        if palavra is not None:
            valor = lexicon[palavra]
            achadas.append(
                {
                    "palavra": palavra,
                    "tipo": "positiva" if valor > 0 else "negativa",
                    "valor": formatar_decimal(valor, 1, sinal=True),
                }
            )
        elif limpa in negacoes or _RE_CONTRACAO_NEGATIVA.search(limpa):
            achadas.append({"palavra": limpa, "tipo": "negacao", "valor": ""})
        elif limpa in intensificadores:
            achadas.append({"palavra": limpa, "tipo": "intensificador", "valor": ""})
    return achadas


def analisar_texto(
    texto: str,
    *,
    analisador=None,
    preprocessar: Optional[Callable[[str], str]] = None,
) -> dict:
    """Analisa um texto (original e pré-processado) e mostra as palavras-chave."""
    analisador = analisador or obter_analisador()
    preprocessar = preprocessar or texto_preprocessado

    original = pontuar(texto, analisador)
    processado_texto = preprocessar(texto)
    processado = pontuar(processado_texto, analisador) if processado_texto else None
    if processado is not None:
        processado["texto"] = processado_texto

    constantes = getattr(analisador, "constants", None)
    palavras = destacar_palavras(
        texto,
        getattr(analisador, "lexicon", {}),
        negacoes=getattr(constantes, "NEGATE", ()),
        intensificadores=getattr(constantes, "BOOSTER_DICT", ()),
    )

    diferenca = None
    if processado is not None:
        diferenca = {
            "muda_classe": processado["classe"] != original["classe"],
            "compound": formatar_decimal(
                processado["compound"] - original["compound"], 4, sinal=True
            ),
        }
    return {
        "original": original,
        "preprocessado": processado,
        "palavras": palavras,
        "diferenca": diferenca,
    }


# --------------------------------------------------------------------------
# Validação no corpus: VADER x nota do CSV
# --------------------------------------------------------------------------
def _acuracia(pares: list[tuple[str, str]]) -> dict:
    """Acerto do VADER só nas notas claras (1, 2, 4, 5): a nota 3 fica de fora."""
    consideradas = [(nota, vader) for nota, vader in pares if nota != NEUTRO]
    acertos = sum(1 for nota, vader in consideradas if nota == vader)
    total = len(consideradas)
    percentual = acertos / total * 100 if total else 0.0
    return {
        "percentual": formatar_percentual(percentual),
        "acertos": formatar_inteiro(acertos),
        "total": formatar_inteiro(total),
    }


def avaliar_corpus(
    textos: Iterable[str],
    notas: Iterable,
    *,
    analisador=None,
    preprocessar: Optional[Callable[[str], str]] = None,
    exemplos: int = 5,
) -> dict:
    """Compara a classe do VADER com a classe da nota (4-5 positivo, 1-2 negativo).

    Devolve a matriz de confusão (linhas = nota, colunas = VADER), a acurácia com
    o texto original e com o pré-processado, a distribuição das classes e
    exemplos em que o VADER discordou fortemente da nota.
    """
    analisador = analisador or obter_analisador()
    preprocessar = preprocessar or texto_preprocessado

    pares_original, pares_processado = [], []
    matriz = {nota: dict.fromkeys(CLASSES, 0) for nota in CLASSES}
    distribuicao = dict.fromkeys(CLASSES, 0)
    divergentes = []
    mudaram = []  # (classe da nota, classe no original, classe no pré-processado)

    for texto, nota in zip(textos, notas):
        classe_nota = classificar_nota(nota)
        if classe_nota is None:
            continue
        texto = str(texto)
        compound = analisador.polarity_scores(texto)["compound"]
        classe_original = classificar_compound(compound)
        compound_proc = analisador.polarity_scores(preprocessar(texto))["compound"]
        classe_proc = classificar_compound(compound_proc)

        pares_original.append((classe_nota, classe_original))
        pares_processado.append((classe_nota, classe_proc))
        matriz[classe_nota][classe_original] += 1
        distribuicao[classe_original] += 1
        if classe_original != classe_proc:
            mudaram.append((classe_nota, classe_original, classe_proc))

        if {classe_nota, classe_original} == {POSITIVO, NEGATIVO}:
            divergentes.append((abs(compound), texto, nota, compound))

    total = len(pares_original)
    linhas_matriz = []
    for classe_nota in CLASSES:
        linha = matriz[classe_nota]
        soma = sum(linha.values())
        acerto = linha[classe_nota] / soma * 100 if soma else 0.0
        linhas_matriz.append(
            {
                "nota": classe_nota,
                "celulas": [
                    {"classe": c, "valor": formatar_inteiro(linha[c]), "diagonal": c == classe_nota}
                    for c in CLASSES
                ],
                "total": formatar_inteiro(soma),
                "acerto": formatar_percentual(acerto) if soma else "—",
            }
        )

    # das que mudaram de classe, em quantas (com nota clara) cada versão acertou
    claras = [(n, o, p) for n, o, p in mudaram if n != NEUTRO]
    efeito_preprocessamento = {
        "total": formatar_inteiro(len(mudaram)),
        "percentual": formatar_percentual(len(mudaram) / total * 100) if total else "—",
        "claras": formatar_inteiro(len(claras)),
        "original_acertou": formatar_inteiro(sum(1 for n, o, _ in claras if n == o)),
        "preprocessado_acertou": formatar_inteiro(sum(1 for n, _, p in claras if n == p)),
    }

    # os mais "convictos" (compound mais longe de zero) primeiro
    divergentes.sort(key=lambda item: -item[0])
    curtos = [d for d in divergentes if len(d[1]) <= 400][:exemplos]

    return {
        "total_textos": formatar_inteiro(total),
        "acuracia_original": _acuracia(pares_original),
        "acuracia_preprocessado": _acuracia(pares_processado),
        "efeito_preprocessamento": efeito_preprocessamento,
        "distribuicao": [
            {
                "classe": classe,
                "total": formatar_inteiro(distribuicao[classe]),
                "percentual": formatar_percentual(distribuicao[classe] / total * 100) if total else "—",
                "largura": round(distribuicao[classe] / total * 100) if total else 0,
            }
            for classe in (POSITIVO, NEUTRO, NEGATIVO)
        ],
        "matriz": linhas_matriz,
        "classes": CLASSES,
        "divergencias": [
            {
                "texto": texto_para_exibir(texto),
                "nota": f"{float(nota):.0f}",
                "compound": formatar_decimal(compound, 4, sinal=True),
                "classe": classificar_compound(compound),
            }
            for _, texto, nota, compound in curtos
        ],
    }


def resumo_sentimento_corpus(df, tamanho_amostra: int = 3000) -> dict:
    """Roda a validação na mesma amostra fixa usada na página do corpus."""
    amostra = df.sample(n=min(tamanho_amostra, len(df)), random_state=42)
    return avaliar_corpus(
        amostra["review/text"].fillna("").tolist(),
        amostra["review/score"].tolist(),
    )
