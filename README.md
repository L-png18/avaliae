# Projeto: avaliações de livros (PLN com Django)

Site em Django que usa o CSV `data/books-15k.csv` (Amazon Book Rating) para as
atividades de pré-processamento de textos e análise de sentimento.

## Como rodar

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
python manage.py runserver 8080
```

Abra http://127.0.0.1:8080/ no navegador. Se a porta der erro de permissão,
troque o número (8001, 9000…).

O comando `nltk.downloader` baixa os recursos do NLTK uma única vez. Se você
esquecer, o site tenta baixar sozinho na primeira vez que abrir a página de
pré-processamento.

## Testes

```bash
python manage.py test
```

Os testes do pré-processamento não precisam do NLTK.

## Estrutura

```
manage.py
config/                       configurações do projeto (settings.py, urls.py)
data/books-15k.csv            base de dados
livros/                       app principal
  services/dataset.py         leitura do CSV e resumo (sem dependência do Django)
  services/preprocessamento.py  limpeza, tokenização, stopwords, lematização
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

## Etapas

1. **Base do projeto** (feita): Django, layout HTML/CSS e leitura do CSV.
2. **Pré-processamento de textos** (em andamento): minúsculas, remoção de pontuação,
   tokenização, stopwords e lematização (NLTK).
3. **Análise de sentimento**: VADER.
4. **Machine Learning**: TF-IDF + Naive Bayes ou SVM.

## Decisões do pré-processamento

- Os textos estão em **inglês**, então as stopwords e o lematizador são do inglês.
- As **negações** (`not`, `no`, `don't`…) são mantidas por padrão: "not good" tem
  sentido oposto a "good", e isso importa na análise de sentimento.
- A lematização tenta cada palavra como verbo, depois substantivo e depois
  adjetivo (`was` → `be`, `books` → `book`, `better` → `good`). É uma
  aproximação simples, sem identificar a classe gramatical.
- `meu_tokenizador` e `remover_stopwords` são as funções do exercício do slide.

## Observações sobre os dados

- `Price` está vazio em cerca de 84% das linhas.
- `User_id` e `profileName` estão vazios em cerca de 19% das linhas.
- A coluna `Id` identifica o livro (edição), não a avaliação.
