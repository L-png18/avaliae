"""Testes da análise de sentimento (etapa 3).

Usam um analisador "de mentira" no lugar do VADER, então rodam sem o NLTK:
    python manage.py test
Os testes com o VADER de verdade só rodam se o ``vader_lexicon`` estiver baixado.
"""
import unittest

import pandas as pd

from livros.services.dataset import sortear_avaliacao_com_nota
from livros.services.sentimento import (
    LIMIAR,
    NEGATIVO,
    NEUTRO,
    POSITIVO,
    analisar_texto,
    avaliar_corpus,
    classificar_compound,
    classificar_nota,
    destacar_palavras,
    formatar_decimal,
    pontuar,
    texto_para_exibir,
)


class AnalisadorFalso:
    """Imita o SentimentIntensityAnalyzer: soma a valência das palavras do léxico."""

    lexicon = {"good": 1.9, "bad": -2.5, ":(": -1.9}

    class constants:
        NEGATE = {"not", "wasn't"}
        BOOSTER_DICT = {"very": 0.293}

    def polarity_scores(self, texto):
        palavras = texto.lower().replace("!", " ").split()
        soma = sum(self.lexicon.get(p, 0) for p in palavras)
        compound = max(-1.0, min(1.0, soma / 4))
        pos, neg = max(compound, 0), max(-compound, 0)
        return {"neg": neg, "neu": 1 - pos - neg, "pos": pos, "compound": compound}


ANALISADOR = AnalisadorFalso()


class ClassesTests(unittest.TestCase):
    def test_limites_do_slide(self):
        self.assertEqual(LIMIAR, 0.05)
        self.assertEqual(classificar_compound(0.0501), POSITIVO)
        self.assertEqual(classificar_compound(0.05), NEUTRO)
        self.assertEqual(classificar_compound(0.0), NEUTRO)
        self.assertEqual(classificar_compound(-0.05), NEUTRO)
        self.assertEqual(classificar_compound(-0.0501), NEGATIVO)

    def test_nota_vira_classe(self):
        self.assertEqual([classificar_nota(n) for n in (5, 4.0)], [POSITIVO, POSITIVO])
        self.assertEqual(classificar_nota(3), NEUTRO)
        self.assertEqual([classificar_nota(n) for n in (2, 1.0)], [NEGATIVO, NEGATIVO])

    def test_nota_ausente_ou_invalida(self):
        for valor in (None, float("nan"), "abc"):
            self.assertIsNone(classificar_nota(valor))

    def test_formatacao_com_virgula(self):
        self.assertEqual(formatar_decimal(0.3641), "0,364")
        self.assertEqual(formatar_decimal(0.85194, 4, sinal=True), "+0,8519")
        self.assertEqual(formatar_decimal(-0.5, 1, sinal=True), "-0,5")


class ExibicaoTests(unittest.TestCase):
    def test_entidades_html_e_tags(self):
        self.assertEqual(texto_para_exibir("J&oacute;zef<br />  was &quot;ok&quot;"), 'Józef was "ok"')


class PontuacaoTests(unittest.TestCase):
    def test_pontuar_texto_positivo(self):
        r = pontuar("good", ANALISADOR)
        self.assertEqual(r["classe"], POSITIVO)
        self.assertEqual(r["compound_texto"], "+0,4750")
        self.assertEqual([b["rotulo"] for b in r["barras"]], ["Negativo", "Neutro", "Positivo"])

    def test_posicao_no_medidor(self):
        self.assertEqual(pontuar("meh", ANALISADOR)["posicao"], 50.0)
        self.assertEqual(pontuar("bad bad", ANALISADOR)["posicao"], 0.0)
        self.assertEqual(pontuar("good good good", ANALISADOR)["posicao"], 100.0)


class DestaqueTests(unittest.TestCase):
    def destacar(self, texto):
        return destacar_palavras(
            texto,
            AnalisadorFalso.lexicon,
            negacoes=AnalisadorFalso.constants.NEGATE,
            intensificadores=AnalisadorFalso.constants.BOOSTER_DICT,
        )

    def test_palavras_do_lexico_com_pontuacao_e_maiusculas(self):
        achadas = self.destacar("Really GOOD!!! but bad.")
        self.assertEqual(
            [(a["palavra"], a["tipo"]) for a in achadas],
            [("good", "positiva"), ("bad", "negativa")],
        )
        self.assertEqual(achadas[0]["valor"], "+1,9")

    def test_emoticon(self):
        achadas = self.destacar("bad :(")
        self.assertEqual([a["palavra"] for a in achadas], ["bad", ":("])

    def test_negacoes_e_intensificadores(self):
        achadas = self.destacar("It wasn't very good, isn't it?")
        self.assertEqual(
            [(a["palavra"], a["tipo"]) for a in achadas],
            [
                ("wasn't", "negacao"),
                ("very", "intensificador"),
                ("good", "positiva"),
                ("isn't", "negacao"),  # qualquer "n't" conta como negação
            ],
        )

    def test_texto_sem_palavras_do_lexico(self):
        self.assertEqual(self.destacar("the book arrived"), [])


class AnaliseDeTextoTests(unittest.TestCase):
    def test_compara_original_e_preprocessado(self):
        r = analisar_texto(
            "The book was very good!!!", analisador=ANALISADOR, preprocessar=lambda t: "bad"
        )
        self.assertEqual(r["original"]["classe"], POSITIVO)
        self.assertEqual(r["preprocessado"]["classe"], NEGATIVO)
        self.assertEqual(r["preprocessado"]["texto"], "bad")
        self.assertTrue(r["diferenca"]["muda_classe"])
        self.assertEqual(r["diferenca"]["compound"], "-1,1000")
        self.assertEqual([p["palavra"] for p in r["palavras"]], ["very", "good"])

    def test_mesma_classe(self):
        r = analisar_texto("good", analisador=ANALISADOR, preprocessar=lambda t: "good good")
        self.assertFalse(r["diferenca"]["muda_classe"])

    def test_nada_sobra_no_preprocessamento(self):
        r = analisar_texto("good", analisador=ANALISADOR, preprocessar=lambda t: "")
        self.assertIsNone(r["preprocessado"])
        self.assertIsNone(r["diferenca"])


class CorpusTests(unittest.TestCase):
    def setUp(self):
        textos = ["good", "bad", "good", "meh", "bad", "good"]
        notas = [5, 1, 1, 3, float("nan"), 2]  # a nota ausente é ignorada
        self.r = avaliar_corpus(
            textos,
            notas,
            analisador=ANALISADOR,
            preprocessar=lambda t: t.replace("bad", ""),  # "bad" some e vira neutro
        )

    def test_total_ignora_nota_ausente(self):
        self.assertEqual(self.r["total_textos"], "5")

    def test_matriz_de_confusao(self):
        valores = {
            linha["nota"]: [c["valor"] for c in linha["celulas"]] for linha in self.r["matriz"]
        }
        # colunas: negativo, neutro, positivo
        self.assertEqual(valores[NEGATIVO], ["1", "0", "2"])
        self.assertEqual(valores[NEUTRO], ["0", "1", "0"])
        self.assertEqual(valores[POSITIVO], ["0", "0", "1"])
        acertos = {linha["nota"]: linha["acerto"] for linha in self.r["matriz"]}
        self.assertEqual(acertos[NEGATIVO], "33,3%")
        diagonais = [c["diagonal"] for c in self.r["matriz"][0]["celulas"]]
        self.assertEqual(diagonais, [True, False, False])

    def test_acuracia_so_com_notas_claras(self):
        self.assertEqual(self.r["acuracia_original"]["percentual"], "50,0%")
        self.assertEqual(self.r["acuracia_original"]["total"], "4")  # a nota 3 fica de fora
        self.assertEqual(self.r["acuracia_preprocessado"]["percentual"], "25,0%")

    def test_efeito_do_preprocessamento(self):
        e = self.r["efeito_preprocessamento"]
        self.assertEqual((e["total"], e["percentual"]), ("1", "20,0%"))
        self.assertEqual((e["claras"], e["original_acertou"], e["preprocessado_acertou"]), ("1", "1", "0"))

    def test_distribuicao(self):
        d = {item["classe"]: (item["total"], item["percentual"]) for item in self.r["distribuicao"]}
        self.assertEqual(d[POSITIVO], ("3", "60,0%"))
        self.assertEqual(d[NEUTRO], ("1", "20,0%"))
        self.assertEqual(d[NEGATIVO], ("1", "20,0%"))

    def test_divergencias_sao_positivo_contra_negativo(self):
        self.assertEqual(
            [(d["nota"], d["classe"]) for d in self.r["divergencias"]],
            [("1", POSITIVO), ("2", POSITIVO)],
        )

    def test_corpus_vazio(self):
        r = avaliar_corpus([], [], analisador=ANALISADOR, preprocessar=lambda t: t)
        self.assertEqual(r["total_textos"], "0")
        self.assertEqual(r["acuracia_original"]["percentual"], "0,0%")
        self.assertEqual(r["distribuicao"][0]["percentual"], "—")


class SorteioComNotaTests(unittest.TestCase):
    def test_devolve_texto_e_nota(self):
        df = pd.DataFrame({"review/text": ["curto", "x" * 200], "review/score": [1.0, 4.0]})
        self.assertEqual(sortear_avaliacao_com_nota(df), ("x" * 200, 4.0))

    def test_sem_coluna_de_nota(self):
        df = pd.DataFrame({"review/text": ["x" * 200]})
        self.assertEqual(sortear_avaliacao_com_nota(df), ("x" * 200, None))


def _vader_disponivel() -> bool:
    """Confere se o léxico já está baixado, sem tentar baixar nada."""
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer

        SentimentIntensityAnalyzer()
        return True
    except Exception:
        return False


@unittest.skipUnless(_vader_disponivel(), "vader_lexicon não está baixado")
class VaderDeVerdadeTests(unittest.TestCase):
    def setUp(self):
        from livros.services.sentimento import obter_analisador

        self.vader = obter_analisador()

    def test_exemplo_do_slide(self):
        r = pontuar("This product is amazing and works perfectly!", self.vader)
        self.assertEqual(r["classe"], POSITIVO)
        self.assertGreater(r["compound"], 0.8)
        self.assertEqual(r["barras"][0]["valor"], "0,000")  # neg

    def test_negacao_inverte(self):
        self.assertEqual(pontuar("good", self.vader)["classe"], POSITIVO)
        self.assertEqual(pontuar("not good", self.vader)["classe"], NEGATIVO)

    def test_maiusculas_e_exclamacao_reforcam(self):
        calmo = pontuar("This book is great.", self.vader)["compound"]
        forte = pontuar("This book is GREAT!!!", self.vader)["compound"]
        self.assertGreater(forte, calmo)

    def test_emoticon_conta(self):
        com = pontuar("The ending was fine :(", self.vader)["compound"]
        sem = pontuar("The ending was fine", self.vader)["compound"]
        self.assertLess(com, sem)


if __name__ == "__main__":
    unittest.main()
