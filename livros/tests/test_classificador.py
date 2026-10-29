"""Testes do Machine Learning (etapa 4).

As partes que não precisam do scikit-learn (rotulagem, métricas, sorteio) sempre
rodam. O treino de verdade usa um dataset pequeno e artificial e só roda se o
scikit-learn estiver instalado:
    python manage.py test
"""
import itertools
import sys
import unittest
from unittest import mock

import pandas as pd

from livros.services.classificador import (
    NEGATIVO,
    POSITIVO,
    ScikitLearnIndisponivel,
    Treinamento,
    _importar_sklearn,
    metricas,
    preparar_dataset,
    prever,
    sortear_avaliacao_de_teste,
    termos_mais_importantes,
    treinar,
)
from livros.tests.test_sentimento import AnalisadorFalso


class PrepararDatasetTests(unittest.TestCase):
    def setUp(self):
        nan = float("nan")
        df = pd.DataFrame(
            {
                "review/text": ["great", "awful", "okay", "great", None, "   ", "fine", "bad"],
                "review/score": [5.0, 1.0, 3.0, 4.0, 5.0, 5.0, nan, 2.0],
            }
        )
        self.dados, self.descartes = preparar_dataset(df)

    def test_fica_so_o_que_serve(self):
        # "great" repetido, "okay" (nota 3), sem texto, texto em branco e sem nota saem
        self.assertEqual(self.dados["texto"].tolist(), ["great", "awful", "bad"])
        self.assertEqual(self.dados["classe"].tolist(), [POSITIVO, NEGATIVO, NEGATIVO])

    def test_conta_o_que_saiu(self):
        self.assertEqual(
            self.descartes, {"total": 8, "sem_dados": 3, "neutras": 1, "repetidas": 1}
        )


class MetricasTests(unittest.TestCase):
    def test_contas_feitas_a_mao(self):
        verdadeiros = [NEGATIVO] * 4 + [POSITIVO] * 6
        previstos = [NEGATIVO, NEGATIVO, NEGATIVO, POSITIVO] + [NEGATIVO] + [POSITIVO] * 5
        m = metricas(verdadeiros, previstos)
        self.assertEqual((m["vp"], m["fn"], m["fp"], m["vn"]), (3, 1, 1, 5))
        self.assertAlmostEqual(m["acuracia"], 0.8)
        self.assertAlmostEqual(m["precisao"], 0.75)  # 3 de 4 chamadas de negativas
        self.assertAlmostEqual(m["revocacao"], 0.75)  # 3 de 4 negativas de verdade
        self.assertAlmostEqual(m["f1"], 0.75)

    def test_baseline_acerta_muito_e_nao_acha_nada(self):
        verdadeiros = [NEGATIVO] * 13 + [POSITIVO] * 87
        m = metricas(verdadeiros, [POSITIVO] * 100)
        self.assertAlmostEqual(m["acuracia"], 0.87)
        self.assertEqual((m["precisao"], m["revocacao"], m["f1"]), (0.0, 0.0, 0.0))

    def test_listas_vazias(self):
        m = metricas([], [])
        self.assertEqual((m["acuracia"], m["f1"]), (0.0, 0.0))


class SortearTesteTests(unittest.TestCase):
    def setUp(self):
        self.t = Treinamento(
            vetorizador=None, naive_bayes=None, svm=None, preprocessar=None, analisador=None,
            textos_teste=["curto", "a" * 100, "b" * 100, "c" * 900],
            classes_teste=[NEGATIVO, NEGATIVO, POSITIVO, POSITIVO],
            resumo={},
        )

    def test_filtra_por_classe_e_tamanho(self):
        primeiro = lambda opcoes: opcoes[0]
        self.assertEqual(sortear_avaliacao_de_teste(self.t, NEGATIVO, escolher=primeiro), ("a" * 100, NEGATIVO))
        self.assertEqual(sortear_avaliacao_de_teste(self.t, POSITIVO, escolher=primeiro), ("b" * 100, POSITIVO))

    def test_sem_texto_de_tamanho_medio_usa_qualquer_um_da_classe(self):
        self.t.textos_teste = ["curto", "c" * 900]
        self.t.classes_teste = [NEGATIVO, POSITIVO]
        texto, classe = sortear_avaliacao_de_teste(self.t, NEGATIVO, escolher=lambda opcoes: opcoes[0])
        self.assertEqual((texto, classe), ("curto", NEGATIVO))


class SemScikitLearnTests(unittest.TestCase):
    def test_erro_amigavel(self):
        # o Python procura "sklearn.svm" direto no cache de módulos, então bloqueamos
        # o pacote e cada submódulo usado (um valor None faz o import falhar)
        bloqueados = {nome: None for nome in list(sys.modules) if nome.split(".")[0] == "sklearn"}
        bloqueados.update(dict.fromkeys(
            ["sklearn", "sklearn.feature_extraction.text", "sklearn.model_selection",
             "sklearn.naive_bayes", "sklearn.svm"]
        ))
        with mock.patch.dict(sys.modules, bloqueados):
            with self.assertRaises(ScikitLearnIndisponivel) as contexto:
                _importar_sklearn()
        self.assertIn("pip install -r requirements.txt", str(contexto.exception))


def _sklearn_disponivel() -> bool:
    try:
        _importar_sklearn()
        return True
    except ScikitLearnIndisponivel:
        return False


def _dataset_artificial() -> pd.DataFrame:
    """80 avaliações positivas e 40 negativas, todas diferentes entre si."""
    positivas = itertools.combinations(
        "good great love wonderful excellent superb lovely brilliant fine nice".split(), 3
    )
    negativas = itertools.combinations(
        "bad awful hate boring terrible dull poor waste".split(), 3
    )
    textos = [" ".join(c) for c in itertools.islice(positivas, 80)]
    textos += [" ".join(c) for c in itertools.islice(negativas, 40)]
    notas = [5.0] * 80 + [1.0] * 40
    return pd.DataFrame({"review/text": textos, "review/score": notas})


@unittest.skipUnless(_sklearn_disponivel(), "scikit-learn não está instalado")
class TreinoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dados, descartes = preparar_dataset(_dataset_artificial())
        cls.treinamento = treinar(
            dados, descartes=descartes, preprocessar=str.lower, analisador=AnalisadorFalso()
        )
        cls.resumo = cls.treinamento.resumo

    def test_divisao_treino_e_teste(self):
        d = self.resumo["dados"]
        self.assertEqual((d["total"], d["treino"], d["teste"]), ("120", "96", "24"))
        self.assertEqual(d["percentual_negativas"], "33,3%")
        self.assertEqual(d["negativas_teste"], "8")  # a divisão mantém a proporção

    def test_quatro_modelos_e_baseline_sem_aprender(self):
        chaves = [m["chave"] for m in self.resumo["modelos"]]
        self.assertEqual(chaves, ["baseline", "vader", "nb", "svm"])
        baseline = self.resumo["modelos"][0]
        self.assertEqual(baseline["f1"], "0,0%")
        self.assertEqual(self.resumo["baseline_acuracia"], "66,7%")  # 16 de 24 positivas

    def test_modelos_aprendem_dados_separaveis(self):
        modelos = {m["chave"]: m for m in self.resumo["modelos"]}
        self.assertEqual(modelos["svm"]["f1"], "100,0%")
        self.assertEqual(modelos["nb"]["f1"], "100,0%")
        self.assertEqual(sum(1 for m in self.resumo["modelos"] if m["melhor"]), 1)

    def test_prever_negativo_e_positivo(self):
        neg = prever(self.treinamento, "bad awful boring")
        self.assertEqual((neg["naive_bayes"]["classe"], neg["svm"]["classe"]), (NEGATIVO, NEGATIVO))
        self.assertGreater(len(neg["termos_negativos"]), 0)
        pos = prever(self.treinamento, "good great love")
        self.assertEqual((pos["naive_bayes"]["classe"], pos["svm"]["classe"]), (POSITIVO, POSITIVO))
        self.assertGreater(len(pos["termos_positivos"]), 0)

    def test_comparacao_com_vader(self):
        self.assertEqual(prever(self.treinamento, "this is bad")["vader"]["binaria"], NEGATIVO)
        self.assertEqual(prever(self.treinamento, "this is good")["vader"]["binaria"], POSITIVO)

    def test_texto_sem_palavras_conhecidas(self):
        r = prever(self.treinamento, "zzz qqq")
        self.assertTrue(r["sem_palavras"])
        self.assertEqual(r["palavras_conhecidas"], 0)
        self.assertNotIn("svm", r)

    def test_termos_mais_importantes(self):
        termos = termos_mais_importantes(self.treinamento.svm, self.treinamento.vetorizador, 5)
        negativos = {t["termo"] for t in termos["negativos"]}
        positivos = {t["termo"] for t in termos["positivos"]}
        self.assertTrue(negativos <= {"bad", "awful", "hate", "boring", "terrible", "dull", "poor", "waste"})
        self.assertTrue(positivos.isdisjoint(negativos))

    def test_so_o_teste_e_sorteavel(self):
        texto, classe = sortear_avaliacao_de_teste(self.treinamento, NEGATIVO, minimo=0, maximo=1000)
        self.assertIn(texto, self.treinamento.textos_teste)
        self.assertEqual(classe, NEGATIVO)


if __name__ == "__main__":
    unittest.main()
