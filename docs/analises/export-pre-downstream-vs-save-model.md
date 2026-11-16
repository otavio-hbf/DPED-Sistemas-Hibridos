# Exports de embeddings: pré-downstream × save_model_embeddings

Este documento registra a diferença entre os dois pontos de exportação do código e as consequências para o ataque de inversão. Referências: [main.py](../../main.py), [exportacao-embeddings-dped.md](../implementação/exportacao-embeddings-dped.md) e [plano-ataque-algen-dped.md](./plano-ataque-algen-dped.md).

## 1. Os dois exports

| Aspecto | `save_dped_student_embeddings` (pré-downstream) | `save_model_embeddings` (pós-run) |
|---|---|---|
| Chamada | `run_method`, somente em `dp-distill`, logo após `distill_embeddings` e antes de `build_dp_compatible_model` e do fine-tuning DP-SGD (`main.py:448`) | Fim de `run_method`, para todos os métodos (`main.py:481`) |
| Diretório | `results/embeddings/<run>/dped_student_pre_downstream/` | `results/embeddings/<run>/` |
| Estado do modelo | Estudante após a destilação com ruído, sem nenhum treino supervisionado | Modelo final, após o treino da tarefa (SST-2) |
| Linhas | Apenas os `token_ids` presentes no conjunto público, ordenados (subconjunto) | Tabela completa `word_embeddings` (30522×768) |
| Arquivos | `token_embeddings.pt/.npy`, `token_ids.npy`, `token_labels.json`, `metadata.txt` | `token_embeddings.pt/.npy`, `*_labels.npy` (classes do SST-2), `label_id_to_name.json`, `metadata.txt` |
| Alinhamento linha → token | Explícito (`token_ids.npy` e `token_labels.json`) | Implícito: linha *i* = ID *i* do vocabulário, sem arquivo de token |
| `metadata.txt` | `stage=after_dped_distillation_before_downstream_finetuning` | Sem campo `stage` |

Os `*_labels.npy` do `save_model_embeddings` são rótulos de classe da tarefa, não palavras. Não servem como alvo lexical.

## 2. Estado atual do disco

- `results/embeddings/glue_sst2_non-private_eps8p0/` contém o export `save_model_embeddings` (`saved_at=2026-11-05`, shape `(30522, 768)`). Não há `token_ids.npy`, `token_labels.json` nem `dped_student_pre_downstream/`.
- Os runs DPED ε∈{8,4,2,1} não estão neste disco. Estão no Drive, em `/content/drive/MyDrive/SIH/privEmbeddings/<data>/`, conforme o notebook.

## 3. Por que o baseline não tem pré-downstream

O método `non-private` não executa `distill_embeddings`. Ele faz fine-tuning direto de `BertForSequenceClassification` (`main.py`, `run_method`). Logo, o único export que ele produz é o pós-run. A decisão de usar o pré-downstream para o baseline exige definir o que é equivalente.

**Decisão tomada: `bert_pretrained`.** A tabela `word_embeddings` do `bert-base-uncased` antes de qualquer treino. Ela é exportada no mesmo formato do pré-downstream, restrita aos mesmos `token_ids` do DPED.

## 4. Consequências

1. **Conjunto de linhas.** O DPED tem um subconjunto de tokens; o baseline seria a tabela completa. Toda comparação deve usar apenas a interseção de `token_ids` entre o baseline e cada ε. Os alvos avaliados precisam ser idênticos em todos os métodos.
2. **O baseline não controla a destilação.** `bert_pretrained` não passa por teachers nem por `distill_embeddings`. A diferença entre DPED e baseline mede o efeito de *toda* a destilação com ruído, e não apenas o efeito do DP. H1 deve ser interpretada assim. Para isolar o efeito do DP, seria preciso um baseline destilado sem ruído. Esse baseline não está escolhido e fica como opção futura.
3. **Não há mistura de treino supervisionado.** O pós-run do `non-private` já foi ajustado a SST-2, então contém sinal da tarefa. Ele não é usado no ataque. Pode aparecer apenas como referência, com essa ressalva.
4. **Tokens com frequência zero.** Em `generate_embeddings` (`models.py`), tokens com `token_freq == 0` recebem a linha pré-treinada do estudante, sem ruído. Esses tokens podem ser idênticos ao `bert_pretrained`. Eles devem ser tratados separadamente (ver P8 no plano).
5. **Rótulo do alvo.** Para o ataque, o alvo é a string em `token_labels.json`. Os rótulos de classe não são usados.
6. **Os dois exports não são intercambiáveis.** Qualquer script de ataque deve ler `token_ids.npy` e recusar pastas que não tenham esse arquivo. Isso evita usar a tabela completa do pós-run por engano.
