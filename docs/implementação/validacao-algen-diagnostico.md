# Estudo de validação do ataque ALGEN (SST-2)

Este documento descreve as alterações e o notebook de diagnóstico criados para validar a capacidade de recuperação do ataque ALGEN no cenário limpo (sem DPED), antes de executar novas grades de privacidade ($\varepsilon$).

## Motivação

Nos resultados anteriores ([metrcs_results.json](../analises/metrcs_results.json)), até mesmo o baseline `bert_pretrained` atingiu métricas baixas (`ROUGE-1 = 0,259`, `BLEU = 1,03` em `n = 1000`), indistinguíveis das condições DPED. Para separar limitações do decodificador congelado, do objeto estático (`word_embeddings`) e do alinhamento linear (`ridge`), foram implementados controles diagnósticos mantendo o decodificador ALGEN congelado (sem retreinamento).

## Alterações no código

### 1. [attacker_dped_sentence.py](../../algen/SIH-ALGEN-main/src/attacker_dped_sentence.py)

- **Novos modos em `--source`:**
  - `bert_contextual`: calcula o *mean-pooling* de `last_hidden_state` do `bert-base-uncased` (ataque original do paper ALGEN) com normalização L2, permitindo comparar diretamente `contextual` vs. `estático` (`bert_pretrained`) nas mesmas frases do SST-2.
  - `oracle_target`: passa `Y_test` (`flan-t5-small` encoder mean-pool) diretamente ao decodificador congelado, medindo o teto absoluto do decodificador no SST-2.
  - `--export_dir` passou a ser opcional quando `--source` não é `dped` (para `bert_pretrained` sem `--export_dir`, usa o vocabulário completo do tokenizer).
- **Opções de alinhamento (sem retreinar o decodificador):**
  - `--center`: centraliza `X_train` e `Y_train` pela média de treino antes do ridge e soma `y_mean` na predição.
  - `--normalize_aligned`: aplica normalização L2 em `X_test_aligned` antes da geração.
- **Métricas diagnósticas no `summary.json`:**
  - `distinct_predictions`: número de strings únicas geradas (detecta colapso de modo).
  - `anisotropy_y_test_mean_cosine`: cosseno médio par-a-par em `Y_test`.
  - `aligned_norm_mean` e `aligned_norm_std`: norma média/desvio de `X_test @ T` antes da pós-normalização.
  - `constant_baseline`: decodificação e métricas do preditor constante `normalize(mean(Y_train))` como piso.

### 2. [direct_static_decode.py](../../algen/SIH-ALGEN-main/src/direct_static_decode.py)

- Controle positivo sem ALGEN (Fase 2 de [plano_melhoria_metodologia.md](plano_melhoria_metodologia.md)): tenta recuperar os tokens diretamente a partir da média estática normalizada via *Orthogonal Matching Pursuit* (OMP) sobre a tabela `word_embeddings` do `bert-base-uncased`.
- Reporta precisão, recall, F1 de conjunto de tokens, `content_word_recall` e `stopword_recall`.

### 3. [validate_algen_study.ipynb](../../notebooks/validate_algen_study.ipynb)

Notebook Colab em 5 etapas (com persistência no Drive em `/content/drive/MyDrive/SIH/algen_validation/<RUN_ID>`):

1. **Etapa 1:** Oráculo do decodificador congelado (`oracle_target`).
2. **Etapa 2:** Controle positivo ALGEN original (`bert_contextual`, `n ∈ {10, 100, 1000}`).
3. **Etapa 3:** Diagnóstico do BERT estático (`bert_pretrained`): escala `n ∈ {1..10000}`, ablação `--center` / `--normalize_aligned` e varredura de `ridge`.
4. **Etapa 4:** Decodificação direta por OMP (`direct_static_decode.py`).
5. **Etapa 5:** Tabela consolidada (`validation_summary.csv`) comparando cada execução contra o piso do preditor constante.

## Validação

- Localmente (sem `torch` instalado): verificação de sintaxe (`py_compile`), estrutura JSON do notebook e testes unitários da lógica OMP em [test_algen_validation.py](../../tests/test_algen_validation.py).
- A execução completa das 5 etapas deve ser feita no Google Colab com GPU.
