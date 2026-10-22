# Projeto: avaliações de livros (PLN com Django)

Site em Django que usa o CSV `data/books-15k.csv` (Amazon Book Rating) para as
atividades de pré-processamento de textos e análise de sentimento.

## Como rodar

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python manage.py runserver
```

Abra http://127.0.0.1:8000/ no navegador.

## Estrutura

```
manage.py
config/                 configurações do projeto (settings.py, urls.py)
data/books-15k.csv      base de dados
livros/                 app principal
  services/dataset.py   leitura do CSV e resumo (sem dependência do Django)
  views.py              páginas: início e base de dados
  urls.py
  templates/livros/     base.html, inicio.html, base_dados.html
  static/livros/        style.css
```

## Etapas

1. **Base do projeto** (feita): Django, layout HTML/CSS e leitura do CSV.
2. **Pré-processamento de textos**: minúsculas, remoção de pontuação, tokenização,
   stopwords e lematização (NLTK).
3. **Análise de sentimento**: VADER.
4. **Machine Learning**: TF-IDF + Naive Bayes ou SVM.

## Observações sobre os dados

- `Price` está vazio em cerca de 84% das linhas.
- `User_id` e `profileName` estão vazios em cerca de 19% das linhas.
- Os textos das avaliações estão em **inglês**; use stopwords e léxico em inglês
  nas próximas etapas.
