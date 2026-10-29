"""Machine Learning supervisionado (etapa 4 do projeto).

Segue o slide "Machine Learning supervisionado":

1. Rotular o dataset: a nota do CSV vira a classe (4 e 5 = positivo, 1 e 2 =
   negativo). A nota 3 fica de fora, porque é ambígua.
2. Converter os textos em vetores: TF-IDF (palavras e pares de palavras) sobre
   o texto pré-processado da etapa 2.
3. Treinar classificadores: Naive Bayes e SVM.
4. Aplicar o modelo para prever se novos comentários são positivos ou negativos.

Cuidados que importam neste dataset:

* Cerca de 87% das avaliações são positivas. Um "modelo" que sempre responde
  "positivo" acerta 87% sem aprender nada. Por isso a avaliação mostra esse
  baseline e olha para a classe rara (negativa) com precisão, revocação e F1.
* Textos repetidos são removidos antes de dividir treino e teste, senão o modelo
  poderia "decorar" respostas que também estão no teste.
* Os hiperparâmetros abaixo foram escolhidos por validação cruzada só no
  conjunto de treino; o conjunto de teste ficou intocado até a avaliação final.
* O Naive Bayes com os valores padrão (``alpha=1``) cai no baseline: a suavização
  é grande demais perto dos pesos pequenos do TF-IDF. Com ``alpha`` pequeno e
  probabilidades a priori iguais ele passa a aprender.

Este módulo não depende do Django. O scikit-learn é importado só quando
necessário, e o pré-processamento e o VADER podem ser trocados pelos parâmetros
``preprocessar`` e ``analisador`` (é o que os testes fazem).
"""
import random
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from .dataset import formatar_inteiro, formatar_percentual
from .sentimento import (
    NEGATIVO,
    POSITIVO,
    classificar_compound,
    classificar_nota,
    formatar_decimal,
    obter_analisador,
    texto_preprocessado,
)

SEMENTE = 42
PROPORCAO_TESTE = 0.2

# Hiperparâmetros (escolhidos por validação cruzada no treino).
# O token_pattern mantém as contrações inteiras ("don't"); o padrão do scikit-learn
# as quebraria em "don" + "t". O desempenho é o mesmo, mas fica coerente com a etapa 2.
PARAMETROS_TFIDF = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "token_pattern": r"[a-z][a-z']+",
}
PARAMETROS_NAIVE_BAYES = {"alpha": 0.03, "fit_prior": False}
PARAMETROS_SVM = {"C": 0.3, "class_weight": "balanced"}

MSG_INSTALAR_SKLEARN = (
    "A biblioteca scikit-learn não está instalada. Rode no terminal: "
    "pip install -r requirements.txt"
)


class ScikitLearnIndisponivel(RuntimeError):
    """O scikit-learn não está instalado."""


def _importar_sklearn():
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import LinearSVC
    except ImportError as erro:
        raise ScikitLearnIndisponivel(MSG_INSTALAR_SKLEARN) from erro
    return np, TfidfVectorizer, train_test_split, MultinomialNB, LinearSVC


# --------------------------------------------------------------------------
# Passo 1: rotular o dataset
# --------------------------------------------------------------------------
def preparar_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Devolve ``(dados, descartes)``; ``dados`` tem as colunas ``texto`` e ``classe``.

    Remove avaliações sem texto ou sem nota, as de nota 3 e os textos repetidos.
    """
    bruto = df[["review/text", "review/score"]].rename(
        columns={"review/text": "texto", "review/score": "nota"}
    )
    total = len(bruto)

    completos = bruto.dropna(subset=["texto", "nota"])
    completos = completos[completos["texto"].astype(str).str.strip() != ""]
    sem_dados = total - len(completos)

    rotulados = completos.assign(classe=completos["nota"].map(classificar_nota))
    polares = rotulados[rotulados["classe"].isin([POSITIVO, NEGATIVO])]
    neutras = len(rotulados) - len(polares)

    unicos = polares.drop_duplicates(subset="texto")
    repetidas = len(polares) - len(unicos)

    dados = unicos[["texto", "classe"]].reset_index(drop=True)
    return dados, {
        "total": total,
        "sem_dados": sem_dados,
        "neutras": neutras,
        "repetidas": repetidas,
    }


# --------------------------------------------------------------------------
# Métricas (feitas à mão, para ficarem fáceis de entender e de testar)
# --------------------------------------------------------------------------
def metricas(verdadeiros, previstos) -> dict:
    """Matriz de confusão e métricas da classe rara (negativa).

    * precisão: das avaliações que o modelo chamou de negativas, quantas eram.
    * revocação: das negativas de verdade, quantas o modelo encontrou.
    * F1: a média harmônica das duas (só é alta se as duas forem altas).
    """
    vp = fn = fp = vn = 0
    for verdadeiro, previsto in zip(verdadeiros, previstos):
        if verdadeiro == NEGATIVO:
            if previsto == NEGATIVO:
                vp += 1
            else:
                fn += 1
        elif previsto == NEGATIVO:
            fp += 1
        else:
            vn += 1

    total = vp + fn + fp + vn
    precisao = vp / (vp + fp) if vp + fp else 0.0
    revocacao = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
    return {
        "vp": vp,
        "fn": fn,
        "fp": fp,
        "vn": vn,
        "acuracia": (vp + vn) / total if total else 0.0,
        "precisao": precisao,
        "revocacao": revocacao,
        "f1": f1,
    }


def _pct(valor: float) -> str:
    return formatar_percentual(valor * 100)


def _linha_modelo(chave: str, nome: str, descricao: str, m: dict) -> dict:
    return {
        "chave": chave,
        "nome": nome,
        "descricao": descricao,
        "acuracia": _pct(m["acuracia"]),
        "precisao": _pct(m["precisao"]),
        "revocacao": _pct(m["revocacao"]),
        "f1": _pct(m["f1"]),
        "f1_valor": m["f1"],
        "f1_largura": round(m["f1"] * 100),
        "vp": formatar_inteiro(m["vp"]),
        "fn": formatar_inteiro(m["fn"]),
        "fp": formatar_inteiro(m["fp"]),
        "vn": formatar_inteiro(m["vn"]),
    }


# --------------------------------------------------------------------------
# Passos 2 e 3: vetorizar (TF-IDF) e treinar
# --------------------------------------------------------------------------
@dataclass
class Treinamento:
    """Tudo o que é preciso guardar depois do treino para avaliar e prever."""

    vetorizador: object
    naive_bayes: object
    svm: object
    preprocessar: Callable[[str], str]
    analisador: object
    textos_teste: list
    classes_teste: list
    resumo: dict


def _rotulo(valor) -> str:
    return NEGATIVO if int(valor) == 1 else POSITIVO


def termos_mais_importantes(svm, vetorizador, quantidade: int = 15) -> dict:
    """Palavras e pares de palavras com maior peso no SVM, para cada classe."""
    pesos = svm.coef_[0]  # peso positivo empurra para "negativo"
    nomes = vetorizador.get_feature_names_out()
    ordem = pesos.argsort()
    return {
        "negativos": [
            {"termo": nomes[i], "peso": formatar_decimal(pesos[i], 2)}
            for i in ordem[::-1][:quantidade]
        ],
        "positivos": [
            {"termo": nomes[i], "peso": formatar_decimal(-pesos[i], 2)}
            for i in ordem[:quantidade]
        ],
    }


def treinar(
    dados: pd.DataFrame,
    *,
    descartes: Optional[dict] = None,
    preprocessar: Optional[Callable[[str], str]] = None,
    analisador=None,
    proporcao_teste: float = PROPORCAO_TESTE,
    semente: int = SEMENTE,
) -> Treinamento:
    """Divide em treino e teste, vetoriza, treina os dois modelos e avalia no teste."""
    np, TfidfVectorizer, train_test_split, MultinomialNB, LinearSVC = _importar_sklearn()
    preprocessar = preprocessar or texto_preprocessado
    analisador = analisador or obter_analisador()

    textos = dados["texto"].astype(str).tolist()
    classes = dados["classe"].tolist()
    y = np.array([1 if c == NEGATIVO else 0 for c in classes])  # 1 = negativo
    processados = [preprocessar(t) for t in textos]

    indices_treino, indices_teste = train_test_split(
        np.arange(len(textos)), test_size=proporcao_teste, stratify=y, random_state=semente
    )

    vetorizador = TfidfVectorizer(**PARAMETROS_TFIDF)
    x_treino = vetorizador.fit_transform([processados[i] for i in indices_treino])
    x_teste = vetorizador.transform([processados[i] for i in indices_teste])

    naive_bayes = MultinomialNB(**PARAMETROS_NAIVE_BAYES).fit(x_treino, y[indices_treino])
    svm = LinearSVC(**PARAMETROS_SVM, random_state=semente).fit(x_treino, y[indices_treino])

    verdadeiros = [classes[i] for i in indices_teste]
    previsoes = {
        "baseline": [POSITIVO] * len(verdadeiros),
        "vader": [
            NEGATIVO
            if classificar_compound(analisador.polarity_scores(textos[i])["compound"]) == NEGATIVO
            else POSITIVO
            for i in indices_teste
        ],
        "nb": [_rotulo(v) for v in naive_bayes.predict(x_teste)],
        "svm": [_rotulo(v) for v in svm.predict(x_teste)],
    }
    resultados = {chave: metricas(verdadeiros, prev) for chave, prev in previsoes.items()}

    modelos = [
        _linha_modelo("baseline", "Sempre \"positivo\"", "Não aprende nada: responde a classe mais comum.", resultados["baseline"]),
        _linha_modelo("vader", "VADER (etapa 3)", "Sem treino. Negativo se a regra do slide disser negativo; neutro conta como positivo.", resultados["vader"]),
        _linha_modelo("nb", "Naive Bayes", "TF-IDF + MultinomialNB.", resultados["nb"]),
        _linha_modelo("svm", "SVM", "TF-IDF + LinearSVC com classes balanceadas.", resultados["svm"]),
    ]
    melhor = max(modelos, key=lambda linha: linha["f1_valor"])["chave"]
    for linha in modelos:
        linha["melhor"] = linha["chave"] == melhor

    total = len(textos)
    negativas = int(y.sum())
    descartes = descartes or {}
    resumo = {
        "dados": {
            "total": formatar_inteiro(total),
            "treino": formatar_inteiro(len(indices_treino)),
            "teste": formatar_inteiro(len(indices_teste)),
            "percentual_negativas": formatar_percentual(negativas / total * 100) if total else "—",
            "negativas_teste": formatar_inteiro(resultados["baseline"]["vp"] + resultados["baseline"]["fn"]),
            "vocabulario": formatar_inteiro(len(vetorizador.vocabulary_)),
            "descartes": {chave: formatar_inteiro(valor) for chave, valor in descartes.items()},
        },
        "modelos": modelos,
        "baseline_acuracia": modelos[0]["acuracia"],
        "melhor_nome": next(linha["nome"] for linha in modelos if linha["melhor"]),
        "termos": termos_mais_importantes(svm, vetorizador),
        "configuracao": {
            "tfidf": "palavras e pares de palavras (1 a 2-gramas), ignorando termos que aparecem em menos de 2 avaliações",
            "naive_bayes": f"alpha = {PARAMETROS_NAIVE_BAYES['alpha']}, probabilidades a priori iguais".replace(".", ","),
            "svm": f"C = {PARAMETROS_SVM['C']}, classes balanceadas".replace(".", ","),
        },
    }
    return Treinamento(
        vetorizador=vetorizador,
        naive_bayes=naive_bayes,
        svm=svm,
        preprocessar=preprocessar,
        analisador=analisador,
        textos_teste=[textos[i] for i in indices_teste],
        classes_teste=verdadeiros,
        resumo=resumo,
    )


# --------------------------------------------------------------------------
# Passo 4: aplicar o modelo a um comentário novo
# --------------------------------------------------------------------------
def prever(treinamento: Treinamento, texto: str, *, quantidade_termos: int = 5) -> dict:
    """Classifica um comentário com os dois modelos e compara com o VADER."""
    processado = treinamento.preprocessar(texto)
    x = treinamento.vetorizador.transform([processado])

    compound = treinamento.analisador.polarity_scores(str(texto))["compound"]
    classe_vader = classificar_compound(compound)
    resultado = {
        "processado": processado,
        "palavras_conhecidas": int(x.nnz),
        "vader": {
            "classe": classe_vader,
            "binaria": NEGATIVO if classe_vader == NEGATIVO else POSITIVO,
            "compound_texto": formatar_decimal(compound, 4, sinal=True),
        },
    }
    if x.nnz == 0:
        # nenhuma palavra do texto existe no vocabulário do modelo: não há o que prever
        resultado["sem_palavras"] = True
        return resultado

    prob_negativo = float(treinamento.naive_bayes.predict_proba(x)[0, 1])
    margem = float(treinamento.svm.decision_function(x)[0])
    resultado["naive_bayes"] = {
        "classe": NEGATIVO if prob_negativo >= 0.5 else POSITIVO,
        "prob_negativo": _pct(prob_negativo),
        "prob_positivo": _pct(1 - prob_negativo),
        "prob_negativo_largura": round(prob_negativo * 100),
    }
    resultado["svm"] = {
        "classe": NEGATIVO if margem > 0 else POSITIVO,
        "margem": formatar_decimal(margem, 2, sinal=True),
    }

    # Quanto cada termo do comentário empurrou o SVM (peso x valor TF-IDF)
    nomes = treinamento.vetorizador.get_feature_names_out()
    contribuicoes = [
        (float(valor * treinamento.svm.coef_[0][indice]), nomes[indice])
        for indice, valor in zip(x.indices, x.data)
    ]
    contribuicoes.sort()
    resultado["termos_negativos"] = [
        {"termo": termo, "peso": formatar_decimal(peso, 2, sinal=True)}
        for peso, termo in reversed(contribuicoes)
        if peso > 0
    ][:quantidade_termos]
    resultado["termos_positivos"] = [
        {"termo": termo, "peso": formatar_decimal(peso, 2, sinal=True)}
        for peso, termo in contribuicoes
        if peso < 0
    ][:quantidade_termos]
    return resultado


def sortear_avaliacao_de_teste(
    treinamento: Treinamento,
    classe: Optional[str] = None,
    minimo: int = 80,
    maximo: int = 500,
    escolher: Callable = random.choice,
) -> tuple[str, str]:
    """Sorteia uma avaliação do conjunto de TESTE (que o modelo nunca viu).

    Devolve ``(texto, classe_verdadeira)``. Com ``classe`` filtra por positivo ou negativo.
    """
    pares = list(zip(treinamento.textos_teste, treinamento.classes_teste))
    if classe:
        pares = [par for par in pares if par[1] == classe]
    medios = [par for par in pares if minimo <= len(par[0]) <= maximo]
    return escolher(medios or pares)
