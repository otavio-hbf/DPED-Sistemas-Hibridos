# Calibracao do ruido na liberacao de embeddings

## Objetivo e escopo

Este plano implementa uma etapa de privacidade para os embeddings de lookup do Student, antes de qualquer fine-tuning downstream. A unidade protegida e um exemplo do conjunto de treino, sob adjacencia por substituicao. O epsilon deste experimento se refere somente a liberacao da tabela de embeddings; o DP-SGD downstream e uma etapa separada e nao participa da grade abaixo.

## Desenho metodologico

1. Usar todos os IDs do vocabulario fixo do tokenizer. A lista nao pode depender dos exemplos, das frequencias de tokens nem do split privado.
2. Para cada teacher e cada token, limitar a norma L2 do vetor de embedding. O limite `C` e calculado como o percentil 95 das normas das linhas da tabela de embeddings do Student pre-treinado, que e uma referencia publica e independente do treino privado.
3. Como os dados sao divididos em `K` particoes disjuntas e fixas, um exemplo afeta no maximo um teacher. Sob adjacencia por substituicao, a sensibilidade L2 do vetor medio de um token e limitada por `2C/K`.
4. Liberar cada vetor de token por um mecanismo Gaussiano. Compor todas as consultas do vocabulario usando o accountant RDP do Opacus, com `sample_rate=1` e `steps=numero_de_tokens`. O accountant determina o multiplicador de ruido para o epsilon e delta alvo; o desvio-padrao aplicado e `multiplicador * (2C/K)`.
5. Treinar o Student para aproximar a tabela ruidosa. Essa destilacao e pos-processamento da saida privatizada. A etapa de DP-SGD downstream nao sera executada na grade de epsilon de embeddings.
6. Usar entropia do sistema operacional para gerar o ruido, sem reutilizar a seed de reproducibilidade do treino.

## Parametros da primeira execucao

- Grade: `epsilon = [1, 2, 4, 8]`.
- Delta inicial: `1e-5`, mantido constante em toda a grade. Antes do treino, validar `delta < 1/N`, onde `N` e o numero de exemplos privados efetivamente usado. Se a condicao falhar, interromper e exigir um delta menor explicitamente.
- Vocabulario: todos os IDs validos do tokenizer configurado.
- Clipping: percentil 95 das normas da tabela pre-treinada do Student; registrar o limite numerico aplicado em cada artefato.

## Verificacao de viabilidade

Com Opacus 1.4.0, `delta=1e-5` e 30.522 tokens (vocabulario BERT-base), a calibracao RDP retornou estes multiplicadores antes de aplicar a sensibilidade:

| Epsilon | Multiplicador de ruido |
| ---: | ---: |
| 1 | 710 |
| 2 | 376,25 |
| 4 | 202,5 |
| 8 | 111,40625 |

O desvio-padrao efetivo e esse multiplicador vezes `2C/K`. Os valores indicam risco de grande degradacao dos vetores. Nao reduzir o conjunto de tokens, alterar `C` com dados privados ou diminuir o ruido para melhorar a utilidade sem uma mudanca metodologica explicita e documentada.

## Execucao e interpretacao

A varredura reutiliza um ensemble de teachers treinado uma vez e produz uma tabela ruidosa e um Student por epsilon, para controlar as demais condicoes. Cada tabela, considerada isoladamente, usa o epsilon correspondente. Se varias tabelas da grade forem liberadas em conjunto, a garantia conjunta exige compor todos os mecanismos; nao declarar que o conjunto das quatro saidas tem epsilon 1, 2, 4 ou 8.

Exemplo para SST-2:

```powershell
python main.py --embedding_privacy_sweep --dataset glue --task sst2 --delta 1e-5
```

A garantia cobre a influencia de um exemplo de treino na tabela liberada. Ela nao garante que um ataque jamais recupere um token ou informacao que ja seja publica. A avaliacao deve medir diretamente a qualidade dos embeddings e o sucesso do ataque de inversao.

## Criterios de validacao

- Testar clipping por vetor, validacao de epsilon/delta e aumento do multiplicador quando epsilon diminui.
- Verificar que o accountant compose exatamente uma consulta por ID do vocabulario e que a lista de IDs independe do dataset.
- Verificar que as exportacoes incluem IDs e labels alinhados para todo o vocabulario e metadados de epsilon, delta, `C`, sensibilidade, multiplicador e desvio-padrao.
- Executar a grade sem DP-SGD downstream e registrar a configuracao e os resultados do ataque.

## Implementacao e validacao

Implementado em `privacy.py`, `models.py` e `main.py`. A opcao `--embedding_privacy_sweep` treina um ensemble uma vez e produz, sem DP-SGD downstream, uma exportacao para cada epsilon da grade. A lista de IDs vem do vocabulario completo do tokenizer; o ruido usa entropia do sistema operacional e nao e afetado por `--seed`.

O argumento `--seed` e preservado em `Config` e usado para inicializar cada Student no sweep. Ele controla a reproducibilidade do treino, mas nao a amostragem do ruido Gaussiano.

O sweep informa o inicio/conclusao do treino e agregacao dos teachers, o inicio e a conclusao de cada liberacao de epsilon (`[1/4]` a `[4/4]`) e o caminho do export correspondente. A destilacao exibe uma barra interna de progresso; as perdas dos teachers permanecem omitidas.

O notebook de ataque executa primeiro um sanity test reduzido em CoLA (64 exemplos, 2 teachers, 1 epoca) e valida os quatro exports antes da grade principal SST-2. Essas saidas sao separadas e nao devem ser interpretadas como resultados experimentais.

Os testes unitarios cobrem clipping, validacao de delta, monotonicidade do multiplicador, o epsilon calculado pelo accountant, a origem independente do ruido e os metadados da tabela privatizada. A exportacao foi atualizada para preservar o alinhamento de todos os IDs e labels sem gravar contagens privadas de tokens.

O preflight confirma que a composicao do vocabulario completo pode resultar em ruido muito alto. Isso e um resultado da escolha de release e do budget, nao um erro a contornar: qualquer reducao do vocabulario ou mudanca de sensibilidade exige revisao metodologica e documentacao propria. O teste unitario nao substitui uma execucao completa com teachers, que depende dos pesos e dos dados do experimento.
