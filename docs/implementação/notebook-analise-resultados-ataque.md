# Notebook de análise dos resultados do ataque ALGEN

Arquivo: [analise_resultados_ataque.ipynb](../../notebooks/analise_resultados_ataque.ipynb)

Lê os resultados gravados por [attack_algen_dped.ipynb](../../notebooks/attack_algen_dped.ipynb) (`SIH/privEmbeddings/<data>/results_attack/`) e analisa as métricas em função de **ε** (DPED e baseline `bert_pretrained`) e de **n** (`train_samples`, pares vazados).

## Entrada

`RESULTS_DIR` (padrão `/content/drive/MyDrive/SIH/privEmbeddings/2026-11-16/results_attack`). A leitura é recursiva: qualquer pasta com um `.json` (`summary.json`) e um `.csv` (`sentence_results.csv`) é uma execução. `epsilon`, `source` e `train_samples` vêm do conteúdo dos arquivos. O nome da pasta (`eps<a>p<b>`) só é usado se `epsilon` estiver vazio. A condição `BERT` é identificada por `source == bert_pretrained`.

São excluídas pastas com `smoke` no nome, execuções classificadas como
`smoke` ou `unclassified` e todas as execuções históricas sem classificação na
célula DPED ε=8, n=10. Uma nova execução explicitamente classificada como
`research` nessa célula é aceita. Outras execuções históricas sem classificação
também são excluídas por padrão. Somente após conferir manualmente que não são
smoke, pode-se definir `ALLOW_UNVERIFIED_LEGACY = True` para carregá-las com
aviso; a célula histórica contaminada continua excluída. Nenhum resultado
histórico é recalculado sem os arquivos brutos.
O carregamento também compara hashes de seleção de frases quando presentes e
interrompe a análise se forem divergentes. A ausência desses hashes é
explicitamente sinalizada.

## Saída

Figuras (`figs/`) e tabelas (`tables/`) em `analysis_attack/`, irmã de `results_attack/` (desligue com `SAVE_OUTPUTS = False`).

## Conteúdo

1. Carga e cobertura (execuções × frases por condição e n).
2. Atributos por frase: comprimento, precisão/recall/F1 de unigramas, recall de palavras de conteúdo vs *stopwords*, repetição e **F1 de acaso** (predição de outra frase da mesma execução; `excess_f1 = f1 - f1_chance`).
3. Heatmaps e curvas de todas as métricas do sumário (BLEU, BLEU-1..4, ROUGE-1/2/L, `exact_match`, cosseno).
4. Tendências por n e por ε com IC95% bootstrap; razões de sensibilidade (ε mínimo/máximo e n máximo/mínimo).
5. Cosseno (alinhamento) vs ROUGE; conferência do sumário contra a média do CSV.
6. Boxplots, ECDF, KDE; testes pareados de Wilcoxon por `sentence_id` com Bonferroni (efeito de n e de ε).
7. Tamanho da frase: comprimento da predição vs referência, adaptação ao tamanho, métricas por faixa de tamanho.
8. Nível de palavra: recuperação por tipo (conteúdo/stopword), raridade, posição e tamanho; palavras mais recuperadas e mais "alucinadas".
9. Diversidade e colapso de modo (vocabulário previsto, concentração nas 20 palavras mais usadas, predições distintas).
10. Dificuldade por frase (frase × configuração, clustermap, fatores de dificuldade) e exemplos de reconstrução.
11. Trade-off: inclinação por década de n, regressão `métrica ~ log10(n) + log2(ε) + interação`, decomposição de variância (frase, n, ε), n mínimo para atingir limiares de ROUGE-1.
12. DPED vs BERT: Δ ROUGE-1 pareado com IC, taxa de vitória e razão.
13. Achados automáticos e exportação.

## Limites

- Há uma única semente (0) e um único ridge (1.0). Os ICs e testes refletem a variação das frases de teste (200 por execução, as mesmas em todas as execuções), não a variação da escolha dos pares de treino.
- `FOCUS_COND` e `FOCUS_N` (seção 7 e seguintes) escolhem a configuração dos gráficos detalhados.
- Validado localmente só com dados sintéticos no mesmo formato. Rodar no Colab com os dados reais é a validação final.
