# Notebook de execução do ataque ALGEN sobre DPED

Arquivo: [attack_algen_dped.ipynb](../../notebooks/attack_algen_dped.ipynb)

Sequência de execução no Colab, no padrão de [dped_medium.ipynb](../../notebooks/dped_medium.ipynb):

1. **Montagem do Drive e clone do repositório.** `git pull` se o repositório já existir, senão `git clone`. Depois `pip install -r requirements.txt`.
2. **Export pré-downstream com `--skip_downstream`.** Os comandos de export pulam a task downstream (DP-SGD, teste e export pós-downstream). O atacante só usa o export pré-downstream, então o treino downstream não é necessário. O re-export grava `token_freq.npy`, necessário ao atacante na opção (c) da máscara de X.

3. **Smoke test.** `attacker_dped_sentence.py` com 10 amostras de treino e 10 de teste. Valida o checkpoint, o export e a escrita dos CSVs.
4. **Grade de ataque.**
   - DPED: ε × `train_samples` ∈ {1, 10, 30, 100, 300, 500, 1000} × ridge. Saída em `results/attack/dped_eps<tag>/n<N>_r<R>/`.
   - Baseline `bert_pretrained`: mesma grade de `train_samples` e ridge. Usa o export de ε=8.0 para o conjunto de tokens e as frequências. Saída em `results/attack/bert_pretrained/`.
5. **Cópia para o Drive.** `results/attack/` vai para `SIH/privEmbeddings/<data>/results_attack/`.

## Decisões e limites

- `ALGEN_CKPT` precisa ser ajustado para a pasta do checkpoint do decodificador ALGEN no Drive. O checkpoint é congelado (sem retreino, ver P5).
- O ridge está fixo em `1.0`. A escolha por validação ou fixa é a decisão P6, ainda em aberto. Ajustar `RIDGE_VALUES` depois dessa decisão.
- A grade roda por `subprocess` com os mesmos argumentos de linha de comando, em vez de uma célula `!python` por execução. Assim a grade completa cabe em uma célula.
- O CSV do atacante usa a coluna `source`, e não `method` como descrito no plano. Não há colunas BLEU no CSV atual, só ROUGE e cosseno. Esses dois pontos precisam ser alinhados antes da análise final.
- Nada disso foi executado localmente (sem `torch`, `transformers` e `datasets` no ambiente). A validação é o smoke test no Colab.
