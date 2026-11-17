# Exportacao dos embeddings do estudante apos DPED

Data: 2026-11-17
Escopo: persistencia dos vetores do Student do metodo `dp-distill` para analise de inversao

## Motivacao

O export anterior era executado ao final de `run_method()`, depois da avaliacao e do fine-tuning downstream. Alem disso, os arquivos `train_labels.npy`, `eval_labels.npy` e `test_labels.npy` guardam classes da tarefa (por exemplo, sentimento no SST-2); eles nao identificam as palavras associadas as linhas da matriz de embeddings.

Para estudar a inversao dos vetores produzidos pela destilacao DPED, e necessario preservar o estado do Student imediatamente apos a destilacao e antes que o treinamento downstream altere seus parametros. Cada vetor tambem precisa estar associado ao ID e ao token BERT correspondente.

## Comportamento implementado

No fluxo `dp-distill` de `main.py`, `save_dped_student_embeddings()` e chamado logo depois de `distill_embeddings()` e antes da conversao do modelo e do fine-tuning final. O export salva as linhas correspondentes a todos os IDs do vocabulario fixo do tokenizer; a lista nao depende do conjunto de treino nem da frequencia dos tokens.

O label e o token textual retornado pelo tokenizer BERT configurado. Portanto, pode ser uma palavra inteira, um fragmento WordPiece ou um token especial. O vocabulario completo inclui tokens especiais e o ID de padding. Estes sao vetores estaticos da tabela de lookup, nao representacoes contextualizadas de ocorrencias em frases. Labels de classe e frequencias privadas nao sao incluidos neste snapshot.

## Arquivos gerados

Para uma execucao com `dataset=glue`, `task=sst2`, `method=dp-distill` e `epsilon=8.0`, o diretorio e:

```text
results/embeddings/glue_sst2_dp-distill_eps8p0/dped_student_pre_downstream/
```

| Arquivo | Conteudo |
| --- | --- |
| `token_embeddings.npy` | Matriz NumPy dos vetores do Student; cada linha corresponde ao mesmo indice nos arquivos de IDs e labels. |
| `token_embeddings.pt` | A mesma matriz em formato PyTorch. |
| `token_ids.npy` | Todos os IDs do vocabulario do tokenizer em ordem crescente. |
| `token_labels.json` | Tokens BERT em ordem correspondente a `token_ids.npy` e as linhas de `token_embeddings`. |
| `metadata.txt` | Etapa, representacao, alinhamento e parametros de privacidade da liberacao: epsilon, delta, limite, sensibilidade e ruido. |

O export legado no diretorio da execucao continua sendo gravado depois do treinamento downstream. Ele representa o estado final do modelo; seus arquivos de labels por split continuam representando classes da tarefa e nao devem ser usados como labels lexicais para estes vetores.

### Execucao sem task downstream (`--skip_downstream`)

Com `--skip_downstream` (somente `--method dp-distill`, execucao unica), `run_method()` retorna logo apos o export pre-downstream. Nao ha fine-tuning DP-SGD, teste nem export pos-downstream. O treino que gera os vetores (teachers e destilacao) roda por completo.

Para a grade de privacidade de embeddings, usar `--embedding_privacy_sweep`. Esse fluxo treina um ensemble uma vez e grava um artefato pre-downstream para cada epsilon em `{1, 2, 4, 8}`. Cada tabela tem seu epsilon individual; a liberacao conjunta de varias tabelas exige composicao dos mecanismos. A grade nao roda teste downstream nem DP-SGD.

## Validacao

O teste `tests/test_embedding_export.py` verifica que:

- todos os IDs do vocabulario, inclusive padding, estao presentes uma vez e em ordem crescente;
- cada label e associado ao ID correspondente;
- cada linha exportada e o vetor correto da tabela do Student.

Validacao executada no ambiente `.venv`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Resultado atualizado: 16 testes aprovados, incluindo calibracao do accountant, clipping por token e exportacao do vocabulario completo. Nao foi executada uma rodada completa de treino dos teachers nem do ataque ALGEN.

Comando para gerar os quatro exports pre-downstream:

```powershell
python main.py --embedding_privacy_sweep --dataset glue --task sst2 --delta 1e-5
```

## Referencias

- `main.py`: `save_dped_student_embeddings()` e chamada logo apos a destilacao em `run_method()`.
- `models.py`: `distill_embeddings()` e `TeacherEnsemble.generate_embeddings()`.
- `tests/test_embedding_export.py`: teste do alinhamento entre vetor, ID e token.
