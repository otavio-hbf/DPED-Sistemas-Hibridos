# Exportacao dos embeddings do estudante apos DPED

Data: 2026-11-16  
Escopo: persistencia dos vetores do Student do metodo `dp-distill` para analise de inversao

## Motivacao

O export anterior era executado ao final de `run_method()`, depois da avaliacao e do fine-tuning downstream. Alem disso, os arquivos `train_labels.npy`, `eval_labels.npy` e `test_labels.npy` guardam classes da tarefa (por exemplo, sentimento no SST-2); eles nao identificam as palavras associadas as linhas da matriz de embeddings.

Para estudar a inversao dos vetores produzidos pela destilacao DPED, e necessario preservar o estado do Student imediatamente apos a destilacao e antes que o treinamento downstream altere seus parametros. Cada vetor tambem precisa estar associado ao ID e ao token BERT correspondente.

## Comportamento implementado

No fluxo `dp-distill` de `main.py`, `save_dped_student_embeddings()` e chamado logo depois de `distill_embeddings()` e antes da conversao do modelo e do fine-tuning final. O export seleciona os IDs nao-padding encontrados no conjunto publico usado na destilacao, ordena-os e salva as linhas correspondentes da tabela `word_embeddings` do Student.

O label e o token textual retornado pelo tokenizer BERT configurado. Portanto, pode ser uma palavra inteira, um fragmento WordPiece ou um token especial; o padding de ID `0` e excluido, de acordo com o comportamento atual de `TeacherEnsemble.generate_embeddings()`. Estes sao vetores estaticos da tabela de lookup, nao representacoes contextualizadas de ocorrencias em frases. Labels de classe nao sao incluidos neste snapshot.

## Arquivos gerados

Para uma execucao com `dataset=glue`, `task=sst2`, `method=dp-distill` e `epsilon=8.0`, o diretorio e:

```text
results/embeddings/glue_sst2_dp-distill_eps8p0/dped_student_pre_downstream/
```

| Arquivo | Conteudo |
| --- | --- |
| `token_embeddings.npy` | Matriz NumPy dos vetores do Student; cada linha corresponde ao mesmo indice nos arquivos de IDs e labels. |
| `token_embeddings.pt` | A mesma matriz em formato PyTorch. |
| `token_ids.npy` | IDs de token em ordem crescente, usados para selecionar as linhas da matriz completa. |
| `token_labels.json` | Tokens BERT em ordem correspondente a `token_ids.npy` e as linhas de `token_embeddings`. |
| `token_freq.npy` | Número de professores cujo conjunto de dados contém cada token, na mesma ordem de `token_ids.npy`. Tokens com valor 0 têm linha não destilada (pré-treinada). Gravado apenas quando `save_dped_student_embeddings` recebe `token_counts` (executado em `dp-distill`). |
| `metadata.txt` | Etapa do pipeline, tipo de representacao, shape e regra de alinhamento. |

O export legado no diretorio da execucao continua sendo gravado depois do treinamento downstream. Ele representa o estado final do modelo; seus arquivos de labels por split continuam representando classes da tarefa e nao devem ser usados como labels lexicais para estes vetores.

### Execucao sem task downstream (`--skip_downstream`)

Com `--skip_downstream` (somente `--method dp-distill`, execucao unica), `run_method()` retorna logo apos o export pré-downstream. Nao ha fine-tuning DP-SGD, teste nem export pos-downstream. O treino que gera os vetores (teachers e destilacao) roda por completo. Metricas de utilidade ficam vazias (`{}`) e o resumo em `results/<dataset>/results_summary.txt` as registra como `nan`. Combinar a flag com `--run_all_methods` ou `--privacy_utility_tradeoff` gera erro.

## Validacao

O teste `tests/test_embedding_export.py` verifica que:

- padding e removido e IDs repetidos aparecem uma vez;
- IDs sao ordenados de forma deterministica;
- cada label e associado ao ID correspondente;
- cada linha exportada e o vetor correto da tabela do Student.

Validacao executada no ambiente `.venv`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Resultado: 6 testes aprovados. A validacao foi feita com teste unitario do export e testes existentes; nao foi executada uma nova rodada completa de treinamento `dp-distill`.

## Referencias

- `main.py`: `save_dped_student_embeddings()` e chamada logo apos a destilacao em `run_method()`.
- `models.py`: `distill_embeddings()` e `TeacherEnsemble.generate_embeddings()`.
- `tests/test_embedding_export.py`: teste do alinhamento entre vetor, ID e token.
