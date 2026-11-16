# Plano de implementação: ataque de inversão ALGEN sobre embeddings DPED

Objetivo: testar se os embeddings do estudante DPED (ε ∈ {8, 4, 2, 1}) são robustos ao ataque de inversão ALGEN, comparando com um baseline não privado. Referência de hipóteses e métricas: [projeto.md](../relatório/projeto.md).

## 1. Decisões já tomadas

| Tema | Decisão | Motivo |
|---|---|---|
| Nível do ataque (vítima) | **Sentence-level (média das linhas da tabela)**: o vetor X de cada frase é a média L2-normalizada das linhas DPED dos tokens da frase | Mantém o pipeline do ALGEN (`mean_pool` + `F.normalize`, alvo em nível de frase) com pouca adaptação: só muda a origem de X. A exportação DPED é uma tabela estática por token (ver [exportacao-embeddings-dped.md](../implementação/exportacao-embeddings-dped.md) e `save_dped_student_embeddings` em [main.py](../../main.py)). |
| Decodificador | **Checkpoint pré-treinado do ALGEN (Zenodo), congelado** | Retreinar é inviável. Nenhuma decisão depende de retreino; qualquer decisão que dependa dele pode ser revogada. |
| Princípio central | **O ataque é o alinhamento.** O atacante aprende um mapeamento X→Y (ridge) a partir de pares vazados e aplica esse mapeamento a vetores não vistos | Define o que é variável no experimento: quantidade de pares (H3), ε e o baseline. |
| Baseline não privado | **BERT pré-treinado sem treino** (`bert-base-uncased`), exportado no ponto pré-downstream, restrito aos mesmos `token_ids` do DPED | Ver [export-pre-downstream-vs-save-model.md](./export-pre-downstream-vs-save-model.md). O método non-private não tem etapa de destilação. |
| Dados do atacante | **Pares (vetor X da frase vazado, texto da frase)** para N frases de SST-2 train; teste em frases fora do alinhamento | Eixo de H3 (quantidade de frases vazadas). |
| Tokens que compõem X | **Apenas tokens com `token_freq > 0`** (presentes em algum professor). Tokens com `token_freq == 0` ficam fora da média de X | Linhas com `token_freq == 0` são a linha pré-treinada do estudante, sem destilação nem ruído. Incluí-las misturaria embedding não tocado com embedding destilado (ver P9). |

## 2. Alternativas consideradas e diferenças

Estas opções ficaram fora do escopo inicial, mas definem como interpretar os resultados.

| Nível | O que o atacante recebe | Alinhamento com ALGEN | Vantagem | Limitação |
|---|---|---|---|---|
| **Sentence-level média (escolhido)** | Média das linhas da tabela DPED para os tokens de cada frase do SST-2 | Igual ao `attacker_gt.py` (`mean_pool` + `F.normalize`) | Métricas padrão do ALGEN (BLEU/ROUGE em frases); menos adaptação | Testa um bag-of-embeddings estático, não a saída contextual do estudante |
| Token-level (descartado) | Uma linha da tabela, rotulada pela string do token | Cada token vira uma "frase" de 1 palavra | Vazamento lexical direto | Métricas degeneradas; desalinhamento maior com o checkpoint de frases. Não é mais o foco. |
| Contextual | Saída do forward pass do estudante por frase (depois do DPED) | Igual ao sentence-level | Representação mais próxima do que um downstream expõe | Exige novo export (não existe hoje); mais custo de GPU. Opção futura. |

Consequência: o resultado sentence-level com média estática mede quanto a tabela DPED vaza o conteúdo das frases pela composição das suas linhas. Não mede o vetor contextual que um modelo de downstream recebe. Isso deve ser declarado nos resultados.

## 3. Cadeia de pipeline e artefatos

1. **Export DPED** (já existente): `results/embeddings/glue_sst2_dp-distill_eps{8,4,2,1}/` (ou `dped_student_pre_downstream/`). Arquivos: `token_embeddings.npy` (N×768), `token_ids.npy`, `token_labels.json`. **Cobre apenas os tokens do conjunto público** e **não grava `token_freq`** (ver P9).
2. **Export baseline** (a gerar): BERT pré-treinado, mesmos `token_ids` do DPED. O run `non-private` existente é pós-downstream e não é usado.
3. **Decodificador ALGEN** (Zenodo, congelado): encoder alvo `flan-t5-small` (512-d), decoder de texto. Não há retreino. Referência do script original: [exp.py](../../algen/SIH-ALGEN-main/src/exp.py).
4. **Vetor X por frase**: média das linhas (768-d) dos tokens da frase com `token_freq > 0`, L2-normalizada. Frases sem nenhum token nessa condição são descartadas. **Vetor Y por frase**: `get_Y_embeddings_from_tokens` (flan-t5 mean-pool, L2) sobre o texto da frase.
5. **Alinhamento X→Y**: ridge de 768 para 512 via pseudo-inversa (`mapping_X_to_Y_pooled` em [attacker_gt.py](../../algen/SIH-ALGEN-main/src/attacker_gt.py)), ajustado com N pares vazados.
6. **Inversão**: decodificar o vetor mapeado de cada frase de teste (`decode_embeddings`) e comparar com o texto original.
7. **Métricas**: ver seção 5.

## 4. Fases de implementação

### Fase 0: validação de pré-requisitos (bloqueante)
- P1 (resolvido). Baseline = BERT pré-treinado, exportado no formato pré-downstream. Consequências em [export-pre-downstream-vs-save-model.md](./export-pre-downstream-vs-save-model.md).
- P2 (resolvido: carregar direto). O baseline `bert_pretrained` é lido no attacker a partir do `bert-base-uncased`, indexado pelos `token_ids` do export DPED, sem novo export. Escolha do usuário: opção que poupa mais RAM. Carregar só o tensor de embeddings de palavras (30522×768, ~94 MB em float32) e manter apenas as linhas necessárias. Equivalência com o formato pré-downstream: mesmas linhas e mesmos IDs.
- P2 (alternativa descartada). Gerar um export `bert_pretrained` com `token_ids.npy` e `token_labels.json`, usando os mesmos `token_ids` de um run DPED. Confirmar que os runs DPED ε∈{8,4,2,1} existem no Drive com `dped_student_pre_downstream/`. O disco local só tem o export pós-run do non-private.
- P3. O checkpoint do ALGEN (Zenodo) fica no Drive e é carregado pelo Colab. Não há download local. Registrar o nome do modelo alvo e o hash do arquivo em `docs/analises/` quando estiver disponível no Drive.
- P4. Validar a dependência de Opacus e do runtime do Colab (ver [analise-erro-pos-distilacao.md](./analise-erro-pos-distilacao.md)), somente se for necessário gerar novos embeddings DPED.
- P9 (resolvido: opção c). X de cada frase usa só os tokens com `token_freq > 0`, dentro do conjunto público do export.
  - Motivo: a tabela completa (opção a) misturaria linhas destiladas com linhas não tocadas do BERT, iguais entre todos os ε e ao baseline. A opção b reduziria a amostra e poderia enviesá-la.
  - Consequência: Y (texto completo da frase) contém palavras que não entram em X. O ataque não tem acesso a essas palavras pela tabela. Isso deve ser declarado nos resultados e pode subestimar o vazamento.
  - Requisito de código: o export não grava `token_freq`. Decisão: (i) gravar `token_freq.npy` em `save_dped_student_embeddings` e reexportar os runs DPED ε 8/4/2/1 no Colab, sem retreino. Código ainda não alterado; aguarda aprovação para implementar.
  - A máscara `token_freq > 0` é a mesma para todos os ε, porque os professores e o conjunto público não mudam entre ε. Então o baseline `bert_pretrained` usa a mesma máscara.

### Fase 1: preparação dos dados do ataque
- Frases: subconjunto de SST-2 train (e teste separado), com os tokens de cada frase tokenizados pelo BERT tokenizer.
- Cobertura: frases com pelo menos um token de `token_freq > 0` presente no export (P9, opção c). Número de frases resultante deve ser reportado.
- Vetor X de cada frase: média das linhas do export dos seus tokens, L2. Feito para o baseline `bert_pretrained` e para cada ε, sobre as mesmas frases.
- Vetor Y de cada frase: flan-t5 mean-pool sobre o texto da frase, L2.
- Conjunto de alinhamento (pares vazados): N frases amostradas com semente fixa. Tamanhos N = 1, 10, 30, 100, 300, 500, 1000, como no `attacker_gt.py`.
- Conjunto de teste: frases fora do conjunto de alinhamento, idênticas entre métodos.
- Alvo de texto: a frase original. Reconstrução do texto a partir dos tokens só para o teste de sanidade (P7).

### Fase 2: adaptação do attacker para sentence-level
- Criar um script novo (não alterar o `attacker_gt.py` original): `algen/SIH-ALGEN-main/src/attacker_dped_sentence.py`.
- Reutilizar `DecoderInference`, `mapping_X_to_Y_pooled`, `get_Y_embeddings_from_tokens` e `decode_embeddings` sem mudá-los. A única adaptação é o carregamento de X a partir do export DPED.
- Entrada: diretório de export + checkpoint do decodificador + tamanho do atacante + ridge + semente.
- Saída: um CSV por execução com `epsilon`, `method`, `train_samples`, `ridge`, `seed`, `sentence_id`, `prediction`, `cosine`, `bleu1`, `bleu2`, `rouge1`, `rougeL`.
- Documentar em `docs/implementação/` (requisito do repositório).

### Fase 3: experimento principal (H1, H2)
- Rodar o ataque para baseline não privado e para DPED ε∈{8,4,2,1}, com tamanho de atacante fixo (escolher um, por exemplo 1000).
- Escolher um ridge fixo pelo baseline (validação no conjunto de alinhamento, não no teste).

### Fase 4: sweep de quantidade do atacante (H3)
- Variar `train_samples` ∈ {1, 10, 30, 100, 300, 500, 1000} para cada método.
- Repetir com 3 sementes para ter variância.

### Fase 5: análise
- Tabela por (método, ε, tamanho do atacante) com as métricas.
- Teste de H1: DPED < non-private em cosseno e exact match.
- Teste de H2: tendência monotônica com ε (maior ε, mais vazamento).
- Teste de H3: curva de vazamento versus tamanho do atacante.
- Gráficos em `results/`.

### Fase 6: documentação
- Atualizar `docs/implementação/exportacao-embeddings-dped.md` com o novo consumidor (ataque).
- Registrar resultados em `docs/analises/`.

## 5. Métricas

- **BLEU-1/2** e **ROUGE-1/L**, calculados como no ALGEN (`eval_metrics.py`, `evaluate.load("sacrebleu")`), conforme [projeto.md](../relatório/projeto.md).
- **Cosseno** entre o vetor predito e o vetor Y da frase original (métrica de alinhamento).
- Médias por (método, ε, N), com a semente como fator de variância.

## 6. Pendências e perguntas em aberto

- P5 (resolvido). Sem retreino. O decodificador do Zenodo é congelado.
- P6: o ridge é escolhido por validação ou fixo?
- P7: a reconstrução do texto a partir dos tokens WordPiece (teste de sanidade) usa `tokenizer.decode`? Os textos de referência são as frases originais.
- P8 (resolvido: excluídos de X, ver P9). Tokens com `token_freq == 0` ficam fora da média de X.

## 7. Critérios de validação

- Teste de sanidade: o ataque sobre o baseline `bert_pretrained` deve gerar BLEU/ROUGE acima do controle. Se não, o pipeline está quebrado e o resultado DPED não tem significado.
- Teste de controle: vetores X aleatórios com a mesma norma devem dar BLEU/ROUGE próximos de zero.
- Cada execução salva seus parâmetros e a semente.

## 8. Backlog

- **Notebooks de execução dos experimentos.** Gerar notebooks `.ipynb` em `notebooks/` para rodar os experimentos no Colab, no mesmo padrão de [dped_medium.ipynb](../../notebooks/dped_medium.ipynb).
  - Requisito: os scripts devem aceitar argumentos de linha de comando, usados no notebook como `!python main.py --dataset glue --task sst2 ...`.
  - Referência atual: o notebook chama `main.py --method dp-distill` para ε 8→4→2→1 e copia as saídas para o Drive.
  - Inclui o script do ataque (`attacker_dped_sentence.py`, Fase 2) e a geração do export `bert_pretrained` (P2).
  - Documentar em `docs/implementação/` quando implementado.
  - Status: notebook de sequência criado em [attack_algen_dped.ipynb](../../notebooks/attack_algen_dped.ipynb) (ver [notebook-attack-algen-dped.md](../implementação/notebook-attack-algen-dped.md)). Pendente: executar no Colab e validar o smoke test.
