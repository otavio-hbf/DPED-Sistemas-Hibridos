# Ataque ALGEN sentence-level sobre embeddings DPED

Script: [attacker_dped_sentence.py](../../algen/SIH-ALGEN-main/src/attacker_dped_sentence.py)

Plano de referência: [plano-ataque-algen-dped.md](../analises/plano-ataque-algen-dped.md)

## O que o script faz

- **X (fonte, por frase):** média das linhas de `token_embeddings.npy` dos tokens de conteúdo da frase presentes no vocabulário completo exportado, normalizada em L2. Tokens especiais (`[CLS]`, `[SEP]`, `[PAD]`) são descartados. Frases sem nenhum token restante são excluídas.
- **Y (alvo, por frase):** média do encoder do flan-t5 sobre a frase, normalizada em L2 (mesmo cálculo do ALGEN).
- **Mapeamento:** ridge `X→Y` ajustado no conjunto de treino do atacante, via `mapping_X_to_Y_pooled` do ALGEN.
- **Decoder:** checkpoint ALGEN congelado. Não há retreinamento, conforme o princípio do projeto.
- **Métricas:** BLEU-1/2 e ROUGE-1/L de corpus (via `DecoderInference.test`), mais cosseno e ROUGE por frase.

## Decisões de implementação

- O atacante usa `DecoderInference` por herança (`DPEDSentenceDecoder`). Só `initialize_embeddings` é sobrescrito. `source_model_name="random"` é passado apenas para evitar o carregamento de um encoder fonte no `__init__` do pai. Essa sobrescrita substitui o X aleatório, então nenhum vetor aleatório é usado.
- Dados: treino do atacante = amostra aleatória (com `--seed`) de SST-2 `train` entre as frases elegíveis. Teste = as primeiras `--test_samples` frases elegíveis de SST-2 `validation`.
- Referências: como no ALGEN, a referência é o texto decodificado dos tokens do flan-t5, não a frase original. Isso mantém a comparação com o ALGEN.
- Baseline `bert_pretrained` (P2): a tabela é lida direto do `bert-base-uncased`, indexada pelos `token_ids` do export DPED. O mesmo filtro de tokens especiais e a mesma lista fixa de IDs são usados em DPED e no baseline.

## Entradas

- `--export_dir`: pasta de um export DPED com `token_ids.npy` e `token_embeddings.npy`. O script usa todos os IDs exportados, exceto tokens especiais; nao requer frequencias derivadas do treino.
- `--checkpoint_path`: pasta do checkpoint do decoder ALGEN (carregado do Drive).
  - `load_best_model` (em `attacker_gt.py`) lê `training_args_and_best_models.json`. Cada entrada de `best_models` é `(val_loss, val_results, caminho)`, e o caminho do checkpoint é sempre o último item. Antes, o código escolhia o formato pelo texto de `checkpoint_path` (procurava `"yiyic"`). No Colab o caminho é `/content/drive/.../checkpoint`, então o ramo errado era usado e o erro `too many values to unpack` aparecia. Agora o código lê `best_entry[0]` e `best_entry[-1]`, o que funciona para qualquer tamanho de tupla.
    - O caminho gravado no JSON é absoluto e pode ficar desatualizado se o checkpoint for renomeado no Drive (ex.: `checkpoint_epoch_100.pt` no JSON, `checkpoint_epoch_99.pt` no disco). Nesse caso, ajuste o nome no JSON ou no arquivo.
    - `torch.load` usa `weights_only=False`. O checkpoint guarda escalares numpy em `val_results`, e o `weights_only=True` padrão do torch>=2.6 rejeita esse tipo. O arquivo vem do próprio treino, então é confiável.
## Saídas

Em `--output_dir`:
- `sentence_results.csv`: uma linha por frase de teste, com `epsilon`, `source`, `train_samples`, `ridge`, `seed`, `sentence_id`, `prediction`, `reference`, `cosine`, `rouge1`, `rougeL`.
- `summary.json`: métricas de corpus, cosseno médio, dimensões, número de frases elegíveis.

## Exemplo de uso (Colab)

```
!python src/attacker_dped_sentence.py --checkpoint_path <ckpt> --export_dir <export_eps8> --epsilon 8 --train_samples 300 --ridge 0.1 --seed 0 --output_dir <saida>
!python src/attacker_dped_sentence.py --checkpoint_path <ckpt> --export_dir <export_eps8> --source bert_pretrained --train_samples 300 --ridge 0.1 --seed 0 --output_dir <saida_baseline>
```

## Limitações conhecidas

- O ataque é sentence-level sobre a média estática dos vetores por token; não mede diretamente a saída contextual do encoder.
- O ambiente local não tem `torch`. A validação feita localmente foi apenas `py_compile`. A execução completa deve ser verificada no Colab.
- A seleção de ridge (fixo vs. validado, P6) ainda está em aberto no plano.
