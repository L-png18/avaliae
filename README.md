# Projeto: avaliações de livros (PLN com Django)

Site em Django que usa o CSV `data/books-15k.csv` (Amazon Book Rating) para as
atividades de pré-processamento de textos e análise de sentimento.

## Como rodar

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4 vader_lexicon
python manage.py runserver 8080
```

Abra http://127.0.0.1:8080/ no navegador. Se a porta der erro de permissão,
troque o número (8001, 9000…).

O comando `nltk.downloader` baixa os recursos do NLTK uma única vez. Se você
esquecer, o site tenta baixar sozinho na primeira vez que abrir a página de
pré-processamento ou de sentimento.

## Testes

```bash
python manage.py test
```

Os testes não precisam do NLTK nem do scikit-learn. Os que usam o VADER de verdade ou treinam
modelos só rodam se o `vader_lexicon` e o scikit-learn estiverem instalados (senão são pulados).

## Estrutura

```
manage.py
config/                       configurações do projeto (settings.py, urls.py)
data/books-15k.csv            base de dados
livros/                       app principal
  services/dataset.py         leitura do CSV e resumo (sem dependência do Django)
  services/preprocessamento.py  limpeza, tokenização, stopwords, lematização
  services/sentimento.py      análise de sentimento com o VADER e validação com as notas
  services/classificador.py   Machine Learning: TF-IDF, Naive Bayes, SVM, métricas e previsão
  views.py                    páginas do site
  urls.py
  templates/livros/           HTML
  static/livros/              CSS
  tests/                      testes
```

## Páginas

- `/` início, com as etapas do projeto
- `/base-de-dados/` resumo do CSV
- `/pre-processamento/` passo a passo de um comentário (escrito por você ou sorteado do CSV)
- `/pre-processamento/corpus/` efeito do pré-processamento em 3.000 avaliações
- `/sentimento/` VADER num comentário (escrito por você ou sorteado do CSV): classe, `compound`,
  palavras que pesaram e comparação entre o texto original e o pré-processado
- `/sentimento/validacao/` o VADER contra as notas do CSV em 3.000 avaliações (acurácia e matriz de confusão)
- `/machine-learning/` treina Naive Bayes e SVM (TF-IDF) e compara com o VADER e com um baseline, com as palavras
  mais importantes. O primeiro acesso leva cerca de 15 s, porque treina os modelos; depois fica em cache
- `/machine-learning/testar/` classifica um comentário com os dois modelos e o VADER; os sorteios vêm só do
  conjunto de teste

## Etapas

1. **Base do projeto** (feita): Django, layout HTML/CSS e leitura do CSV.
2. **Pré-processamento de textos** (feita): minúsculas, remoção de pontuação,
   tokenização, stopwords e lematização (NLTK).
3. **Análise de sentimento** (feita): VADER.
4. **Machine Learning** (feita): TF-IDF + Naive Bayes e SVM.

## Decisões do pré-processamento

- Os textos estão em **inglês**, então as stopwords e o lematizador são do inglês.
- As **negações** (`not`, `no`, `don't`…) são mantidas por padrão: "not good" tem
  sentido oposto a "good", e isso importa na análise de sentimento.
- A lematização tenta cada palavra como verbo, depois substantivo e depois
  adjetivo (`was` → `be`, `books` → `book`, `better` → `good`). É uma
  aproximação simples, sem identificar a classe gramatical.
- `meu_tokenizador` e `remover_stopwords` são as funções do exercício do slide.

## Decisões da análise de sentimento

- A classe sai do `compound` do VADER, com a regra do slide: `> 0,05` positivo,
  `< -0,05` negativo e o resto neutro.
- O VADER analisa o **texto original** (ele usa `!!!`, MAIÚSCULAS, emoticons e a posição das
  negações). O texto pré-processado também é analisado, só para comparar.
- Para comparar com o CSV, a nota vira classe: 4 e 5 = positivo, 3 = neutro, 1 e 2 = negativo.
  A acurácia ignora a nota 3, que é ambígua.
- Na amostra de 3.000 avaliações, a acurácia é praticamente a mesma com o texto original e com
  o pré-processado (~82%), mas cerca de 8% das avaliações mudam de classe: pré-processar não é
  neutro para o VADER, só não melhora. O VADER acerta bem menos nas avaliações negativas (~40%)
  do que nas positivas (~88%), e o dataset tem muito mais avaliações positivas.

## Decisões do Machine Learning

- **Rótulo:** vem da nota (4 e 5 = positivo, 1 e 2 = negativo). A nota 3 fica fora, assim como os
  textos repetidos (77 entre as avaliações usadas), para um mesmo texto não cair no treino e no teste.
- **Divisão:** 80% treino e 20% teste, estratificada (mantém ~13% de negativas nos dois) e com
  semente fixa (42), então os números das páginas são sempre os mesmos.
- **O desbalanceamento é o ponto central.** 87% das avaliações são positivas, então "sempre
  positivo" já acerta 87% sem aprender nada. Por isso a página mostra esse baseline e avalia as
  negativas com precisão, revocação e F1, e não só a acurácia.
- **Resultado no teste (355 negativas):** baseline F1 0% (acurácia 87,0%), VADER F1 40,3%, Naive Bayes
  F1 59,4% e SVM F1 69,2% (acurácia 91,9%). Com poucas negativas no teste, diferenças de poucos
  pontos entre modelos podem ser sorte; já a distância entre os modelos treinados e o VADER é grande.
- **Naive Bayes:** com `alpha = 1` (padrão) ele cai no baseline, porque a suavização é grande demais
  perto dos pesos pequenos do TF-IDF. Com `alpha = 0,03` e probabilidades a priori iguais, aprende.
- **Hiperparâmetros** (TF-IDF com 1 a 2-gramas, `min_df = 2`; NB `alpha = 0,03`; SVM `C = 0,3` com
  classes balanceadas) foram escolhidos por validação cruzada só no treino. Pares de palavras
  ajudam (capturam `not worth`, `not disappoint`).
- **Texto de entrada:** o TF-IDF usa o texto pré-processado da etapa 2, com um `token_pattern` que
  mantém contrações como `don't`. Num experimento fora do site, texto bruto e pré-processado
  empataram dentro da variação (F1 ≈ 0,69 ± 0,04 contra ≈ 0,67 ± 0,02, validação cruzada).
- **Limite:** os modelos só conhecem positivo e negativo, então um comentário neutro é sempre
  empurrado para um dos lados. O rótulo é a nota do usuário, que nem sempre combina com o texto.

## Observações sobre os dados

- `Price` está vazio em cerca de 84% das linhas.
- `User_id` e `profileName` estão vazios em cerca de 19% das linhas.
- A coluna `Id` identifica o livro (edição), não a avaliação.
