"""Testes do pré-processamento.

Usam stopwords, lematizador e stemmer "de mentira", então rodam sem o NLTK:
    python manage.py test
ou, na raiz do projeto:
    python -m unittest discover -s livros/tests -t .
"""
import unittest

import pandas as pd

from livros.services.dataset import sortear_avaliacao
from livros.services.preprocessamento import (
    NEGACOES,
    ajustar_stopwords,
    estatisticas_corpus,
    executar_pipeline,
    limpar_texto,
    meu_tokenizador,
    remover_stopwords,
    tokenizar,
)

STOPWORDS = {"i", "the", "but", "it", "was", "not"}
LEMAS = {"loved": "love", "books": "book"}


def lematizar(palavra):
    return LEMAS.get(palavra, palavra)


def radicalizar(palavra):
    return palavra[:4]


class LimpezaTests(unittest.TestCase):
    def test_minusculas_e_pontuacao(self):
        self.assertEqual(limpar_texto("Great BOOK!!! Loved it..."), "great book loved it")

    def test_remove_html_url_numeros_e_entidades(self):
        texto = "Read <br /> this: http://x.com/a 10/10 &quot;wow&quot;"
        self.assertEqual(limpar_texto(texto), "read this wow")

    def test_mantem_apostrofo_das_contracoes(self):
        self.assertEqual(limpar_texto("I didn't like it"), "i didn't like it")

    def test_texto_vazio(self):
        self.assertEqual(limpar_texto("  !!! 123 "), "")


class TokenizacaoTests(unittest.TestCase):
    def test_meu_tokenizador_espacos(self):
        self.assertEqual(meu_tokenizador("a b  c"), ["a", "b", "c"])

    def test_meu_tokenizador_delimitador(self):
        self.assertEqual(meu_tokenizador("a,b,,c", ","), ["a", "b", "c"])

    def test_tokenizar_possessivo_e_aspas(self):
        self.assertEqual(tokenizar("book's 'quoted' don't"), ["book", "quoted", "don't"])


class StopwordsTests(unittest.TestCase):
    def test_ignora_maiusculas(self):
        self.assertEqual(remover_stopwords(["The", "Book", "is"], ["the", "is"]), ["Book"])

    def test_negacoes_mantidas_por_padrao(self):
        resultado = ajustar_stopwords({"not", "the", "don't"})
        self.assertEqual(resultado, frozenset({"the"}))

    def test_negacoes_removidas_quando_pedido(self):
        resultado = ajustar_stopwords({"not", "the"}, manter_negacoes=False)
        self.assertEqual(resultado, frozenset({"not", "the"}))

    def test_lista_de_negacoes_tem_o_basico(self):
        self.assertTrue({"not", "no", "don't"} <= NEGACOES)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.resultado = executar_pipeline(
            "I loved the books, but it was not good!",
            stopwords=STOPWORDS,
            lematizar=lematizar,
            radicalizar=radicalizar,
        )

    def test_etapas(self):
        r = self.resultado
        self.assertEqual(r["limpo"], "i loved the books but it was not good")
        self.assertEqual(
            r["tokens"], ["i", "loved", "the", "books", "but", "it", "was", "not", "good"]
        )
        self.assertEqual(r["sem_stopwords"], ["loved", "books", "good"])
        self.assertEqual(r["removidas"], ["i", "the", "but", "it", "was", "not"])
        self.assertEqual(r["lematizados"], ["love", "book", "good"])
        self.assertEqual(r["radicais"], ["love", "book", "good"])

    def test_marca_o_que_mudou_na_lematizacao(self):
        mudou = [par["mudou"] for par in self.resultado["pares"]]
        self.assertEqual(mudou, [True, True, False])

    def test_texto_vazio(self):
        r = executar_pipeline("", stopwords=STOPWORDS, lematizar=lematizar, radicalizar=radicalizar)
        self.assertEqual(r["tokens"], [])
        self.assertEqual(r["pares"], [])


class CorpusTests(unittest.TestCase):
    def test_estatisticas(self):
        resumo = estatisticas_corpus(
            ["Books books BOOK", "I love books"],
            stopwords={"i"},
            lematizar=lambda p: "book" if p.startswith("book") else p,
            radicalizar=lambda p: p.rstrip("s"),
        )
        etapas = {e["nome"]: e for e in resumo["etapas"]}
        tokens = etapas["Tokenização (após a limpeza)"]
        sem_stop = etapas["Sem stopwords"]
        lemas = etapas["Lematização"]

        self.assertEqual(resumo["total_textos"], "2")
        self.assertEqual((tokens["total_tokens"], tokens["vocabulario"]), ("6", "4"))
        self.assertEqual((sem_stop["total_tokens"], sem_stop["vocabulario"]), ("5", "3"))
        self.assertEqual((lemas["total_tokens"], lemas["vocabulario"]), ("5", "2"))
        self.assertEqual((tokens["largura"], sem_stop["largura"], lemas["largura"]), (100, 75, 50))
        self.assertEqual(lemas["reducao"], "50,0%")
        self.assertEqual(
            resumo["mais_frequentes"][0], {"palavra": "book", "frequencia": "4"}
        )

    def test_corpus_vazio(self):
        resumo = estatisticas_corpus(
            [], stopwords=set(), lematizar=lambda p: p, radicalizar=lambda p: p
        )
        self.assertEqual(resumo["total_textos"], "0")
        self.assertEqual(resumo["etapas"][0]["largura"], 0)


class SorteioTests(unittest.TestCase):
    def test_sorteia_texto_de_tamanho_medio(self):
        df = pd.DataFrame({"review/text": ["curto", "x" * 200, "y" * 5000]})
        self.assertEqual(sortear_avaliacao(df), "x" * 200)

    def test_sem_candidatos_usa_qualquer_texto(self):
        df = pd.DataFrame({"review/text": ["curto"]})
        self.assertEqual(sortear_avaliacao(df), "curto")


if __name__ == "__main__":
    unittest.main()
