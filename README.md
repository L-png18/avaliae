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

Os testes não precisam do NLTK. Os 4 testes que usam o VADER de verdade só rodam se o
`vader_lexicon` estiver baixado (senão são pulados).

## Estrutura

```
manage.py
config/                       configurações do projeto (settings.py, urls.py)
data/books-15k.csv            base de dados
livros/                       app principal
  services/dataset.py         leitura do CSV e resumo (sem dependência do Django)
  services/preprocessamento.py  limpeza, tokenização, stopwords, lematização
  services/sentimento.py      análise de sentimento com o VADER e validação com as notas
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

## Etapas

1. **Base do projeto** (feita): Django, layout HTML/CSS e leitura do CSV.
2. **Pré-processamento de textos** (feita): minúsculas, remoção de pontuação,
   tokenização, stopwords e lematização (NLTK).
3. **Análise de sentimento** (em andamento): VADER.
4. **Machine Learning**: TF-IDF + Naive Bayes ou SVM.

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

## Observações sobre os dados

- `Price` está vazio em cerca de 84% das linhas.
- `User_id` e `profileName` estão vazios em cerca de 19% das linhas.
- A coluna `Id` identifica o livro (edição), não a avaliação.
