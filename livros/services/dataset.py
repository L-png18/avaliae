"""Leitura e resumo do dataset de avaliações de livros (Amazon Book Rating).

Este módulo não depende do Django: recebe o caminho do CSV como parâmetro,
o que facilita testar e reutilizar nas próximas etapas (pré-processamento).
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Descrição de cada coluna (baseada no slide da atividade).
DESCRICOES = {
    "Id": "Identificador do livro na Amazon (edições diferentes têm Ids diferentes)",
    "Title": "Título do livro",
    "Price": "Preço do livro",
    "User_id": "Identificador do usuário",
    "profileName": "Nome do usuário",
    "review/helpfulness": "Utilidade da avaliação (ex.: 2/3 = 2 de 3 pessoas acharam útil)",
    "review/score": "Nota dada ao livro (1 a 5)",
    "review/time": "Data da avaliação (timestamp Unix)",
    "review/summary": "Resumo da avaliação",
    "review/text": "Texto completo da avaliação",
}

# Nome amigável dos tipos do pandas (versões antigas mostram "object", as novas "str").
TIPOS = {
    "object": "texto",
    "str": "texto",
    "string": "texto",
    "float64": "decimal",
    "int64": "inteiro",
}

TAMANHO_MAXIMO_TEXTO = 160
LINHAS_DA_AMOSTRA = 10


def formatar_inteiro(valor: int) -> str:
    """15000 -> '15.000' (separador de milhar no padrão brasileiro)."""
    return f"{valor:,}".replace(",", ".")


def formatar_percentual(valor: float) -> str:
    """12.3456 -> '12,3%'."""
    return f"{valor:.1f}".replace(".", ",") + "%"


@lru_cache(maxsize=1)
def carregar_dataset(caminho: Path) -> pd.DataFrame:
    """Lê o CSV uma única vez e guarda em cache (o arquivo tem ~14 MB)."""
    df = pd.read_csv(caminho)
    df["data_avaliacao"] = pd.to_datetime(df["review/time"], unit="s")
    return df


def _texto_curto(valor) -> str:
    if pd.isna(valor):
        return "—"
    texto = str(valor).replace("\n", " ")
    if len(texto) > TAMANHO_MAXIMO_TEXTO:
        texto = texto[:TAMANHO_MAXIMO_TEXTO].rstrip() + "…"
    return texto


def resumo_dataset(df: pd.DataFrame) -> dict:
    """Monta tudo o que a página 'Base de dados' precisa exibir."""
    total_linhas = len(df)
    colunas_originais = [c for c in df.columns if c in DESCRICOES]

    colunas = []
    for nome in colunas_originais:
        nulos = int(df[nome].isna().sum())
        colunas.append(
            {
                "nome": nome,
                "descricao": DESCRICOES[nome],
                "tipo": TIPOS.get(str(df[nome].dtype), str(df[nome].dtype)),
                "nulos": formatar_inteiro(nulos),
                "percentual_nulos": formatar_percentual(nulos / total_linhas * 100),
                "tem_nulos": nulos > 0,
            }
        )

    amostra = df.head(LINHAS_DA_AMOSTRA)[colunas_originais]
    linhas_amostra = [
        [_texto_curto(valor) for valor in linha]
        for linha in amostra.itertuples(index=False, name=None)
    ]

    return {
        "total_linhas": formatar_inteiro(total_linhas),
        "total_colunas": len(colunas_originais),
        "livros_unicos": formatar_inteiro(df["Title"].nunique()),
        "usuarios_unicos": formatar_inteiro(df["User_id"].nunique()),
        "primeira_avaliacao": df["data_avaliacao"].min().strftime("%d/%m/%Y"),
        "ultima_avaliacao": df["data_avaliacao"].max().strftime("%d/%m/%Y"),
        "colunas": colunas,
        "cabecalho_amostra": colunas_originais,
        "linhas_amostra": linhas_amostra,
    }


def sortear_avaliacao_com_nota(df: pd.DataFrame, minimo: int = 80, maximo: int = 500):
    """Sorteia uma avaliação (de tamanho médio) e devolve ``(texto, nota)``.

    A nota vem de ``review/score`` e é ``None`` se a coluna não existir.
    """
    textos = df["review/text"].fillna("")
    candidatos = textos[textos.str.len().between(minimo, maximo)]
    if candidatos.empty:
        candidatos = textos
    indice = candidatos.sample(n=1).index[0]
    nota = df.loc[indice, "review/score"] if "review/score" in df.columns else None
    return str(textos.loc[indice]), nota


def sortear_avaliacao(df: pd.DataFrame, minimo: int = 80, maximo: int = 500) -> str:
    """Sorteia o texto de uma avaliação (de tamanho médio, boa para visualizar)."""
    return sortear_avaliacao_com_nota(df, minimo, maximo)[0]
