# Observacao: privacidade e destilacao de embeddings no `dp-distill`

Data da observacao: 2026-11-15
Escopo: comparacao entre a metodologia descrita em `docs/dped_multi_layer_noise_distillation_for_privacy_preserving_text_embeddings.md`, o fluxo implementado no repositorio e o resultado SST-2 fornecido no notebook.

> **Analise historica, anterior a 2026-11-17.** As descricoes do mecanismo de embeddings abaixo registram o estado do codigo analisado naquela data e nao descrevem a implementacao atual. A calibracao por epsilon, o clipping, a composicao RDP e a liberacao do vocabulario completo estao documentados em [`calibracao-ruido-embeddings.md`](../implementação/calibracao-ruido-embeddings.md). O sweep atual mede a liberacao pre-downstream sem DP-SGD; conclusoes antigas sobre `teacher_noise`, frequencias e privacidade dessa liberacao nao devem ser aplicadas a ele.

> **Resumo executivo**
>
> O comando `--epsilon 2.0` define o epsilon-alvo usado pelo Opacus no fine-tuning DP-SGD final. Ele nao calibra o ruido empregado antes na agregacao dos embeddings dos teachers. O DP-SGD final ajusta todos os parametros treinaveis do modelo, incluindo a tabela de embeddings do Student, mas isso nao altera retroativamente a garantia de privacidade da etapa anterior.
>
> A destilacao com teachers em particoes disjuntas e agregacao ruidosa pode ser usada para construir um mecanismo de Differential Privacy. A divisao, por si so, nao constitui uma garantia: e necessario limitar a sensibilidade das saidas, calibrar o ruido, contabilizar as consultas e tratar qualquer decisao derivada dos dados privados. A implementacao atual nao demonstra esses requisitos para a etapa de embeddings. Portanto, o epsilon informado no log deve ser interpretado como o epsilon contabilizado para o fine-tuning DP-SGD, nao como uma garantia end-to-end comprovada de `dp-distill`.

## 1. Contexto do experimento observado

O comando analisado foi:

```bash
python main.py --dataset glue --task sst2 --method dp-distill \
  --epsilon 2.0 --num_epochs 5 --batch_size 16 \
  --max_train_samples 16384 --max_eval_samples 4096 \
  --max_test_samples 1024 --max_public_samples 1024 \
  --subset_strategy random --subset_seed 42
```

O log fornecido registra 16.384 exemplos de treino e 1.024 exemplos publicos. Tanto a avaliacao quanto o conjunto chamado de teste tinham 872 exemplos: o split de validacao do SST-2 contem menos exemplos do que o limite 1.024, e o carregador usa essa mesma validacao como avaliacao e teste (`data_loader.py`, funcao `load_glue_dataset`).

Metricas finais observadas:

| Metrica | Resultado |
| --- | ---: |
| Acuracia | 0,5092 |
| Precisao | 0,5092 |
| Recall | 1,0000 |
| F1 | 0,6748 |

O log tambem registra as mesmas metricas de avaliacao em todas as cinco epocas. A perda media do fine-tuning DP-SGD aumenta de aproximadamente 0,77 para 1,29. Para o alvo `epsilon=2.0`, o accountant do Opacus reporta epsilon gasto de aproximadamente 1,77 apos a primeira epoca e 2,00 ao final da quinta.

### Interpretacao das metricas

O codigo calcula metricas binarias para SST-2, usando a classe positiva como classe positiva (`utils.py`, funcao `compute_metrics`). Com 872 exemplos e acuracia `0,509174...`, o numero de exemplos positivos e compativel com 444. A combinacao de recall 1, precisao igual a acuracia e desempenho abaixo de 1 implica que o modelo classificou os exemplos negativos como positivos: o comportamento e compativel com prever positivo para todos os exemplos.

Isso mostra um colapso da **classificacao final** para uma unica classe. Nao permite, por si so, atribuir a causa a embeddings de baixa qualidade. O gargalo pode estar nos embeddings, na destilacao, na inicializacao do classificador, no fine-tuning ruidoso ou na interacao entre essas etapas. Para separar essas hipoteses, e necessario avaliar a representacao antes do DP-SGD, por exemplo com uma sondagem linear com embeddings congelados e uma matriz de confusao.

O resultado chamado de teste nao e independente da avaliacao: para tarefas GLUE com split `validation`, `data_loader.py` define `test_dataset = eval_dataset`. Alem disso, o melhor checkpoint e escolhido pela acuracia de avaliacao. Assim, as metricas finais desse fluxo nao devem ser apresentadas como desempenho em um conjunto de teste independente.

## 2. Metodologia descrita no documento-base

O documento-base propoe:

1. Dividir os dados privados entre teachers disjuntos.
2. Consultar os teachers com exemplos nao rotulados.
3. Agregar representacoes intermediarias e saidas dos teachers.
4. Proteger essas agregacoes com ruido calibrado.
5. Treinar o Student apenas com sinais privatizados.

A analise de privacidade do documento depende de hipoteses adicionais: limitar a sensibilidade das representacoes (por exemplo, com clipping), calibrar os desvios-padrao do ruido a essa sensibilidade e ao budget de privacidade, e compor o custo de todas as consultas. O texto tambem descreve um mecanismo de consenso/abstencao para palavras raras.

Em termos gerais, treinar cada teacher sem DP nao impede automaticamente uma abordagem PATE: a protecao pode vir das respostas agregadas e privatizadas, desde que o mecanismo seja corretamente limitado, calibrado e contabilizado. A existencia de particoes disjuntas so limita quantos teachers podem ser afetados por um registro; ela nao determina, sozinha, a sensibilidade das respostas nem o ruido necessario.

## 3. Fluxo efetivamente implementado

O ramo `dp-distill` de `main.py` executa, nesta ordem:

1. Cria o ensemble e treina os teachers.
2. Gera e destila embeddings no Student.
3. Converte o Student para um modelo de classificacao compatível com Opacus, copiando pesos com nomes e dimensoes correspondentes.
4. Executa DP-SGD se `config.enable_dp` estiver habilitado.
5. Avalia o modelo e salva os embeddings finais.

### 3.1 Treinamento dos teachers

`TeacherEnsemble.train_teachers` treina cada teacher usando `AdamW`, backpropagation convencional e clipping global de gradiente (`models.py`). Esse clipping e uma operacao de otimizacao, nao um limite provado para a sensibilidade da representacao agregada. Essa etapa nao usa o accountant do Opacus.

### 3.2 Geracao e destilacao dos embeddings

Em `TeacherEnsemble.generate_embeddings`, o codigo:

- obtem vetores da tabela `word_embeddings` de cada BERT teacher;
- calcula a media entre teachers para cada posicao/token;
- adiciona ruido Gaussiano com desvio baseado em `teacher_aggregation_noise` (padrao `0.5`), multiplicado por um fator para tokens raros;
- armazena um vetor por ID de token.

Esses sao vetores da camada de lookup de palavras, nao as representacoes contextuais intermediarias de uma sentenca descritas no documento-base. A funcao `distill_embeddings` ajusta a tabela de embeddings de palavras do Student por MSE. Nessa etapa, nao ha perda supervisionada de sentimento nem treinamento direto da cabeca classificadora.

Ha diferencas relevantes entre o desenho descrito e o codigo:

- O ruido da agregacao e definido por `teacher_aggregation_noise`/`--teacher_noise`, nao pelo argumento `--epsilon`.
- O codigo nao mostra clipping dos vetores agregados para impor o limite de sensibilidade assumido pela analise do documento-base.
- O codigo nao executa um accountant/composicao para as consultas de embedding.
- A frequencia do token entre particoes de teachers influencia o tratamento do token. Para `token_freq == 0`, o codigo usa o embedding pre-treinado do Student, em vez de uma agregacao ruidosa. Como essa ramificacao depende de uma contagem obtida dos dados de treino, sua contribuicao para a privacidade tambem precisa ser analisada.
- `create_public_dataset` seleciona dados de `dataset["train"]`. Portanto, o conjunto chamado de publico no fluxo atual e uma amostra do split de treino, nao uma fonte comprovadamente publica e independente.

O `StudentModel` contem injecao de ruido em seu `forward`, mas a destilacao acima acessa diretamente os pesos de `word_embeddings` e calcula MSE, sem usar esse `forward`. Antes do fine-tuning, o modelo e convertido para `BertForSequenceClassification`, que nao usa o `forward` customizado do `StudentModel`. Assim, essas injecoes em camadas do `StudentModel` nao substituem uma garantia formal para a agregacao dos teachers nesse fluxo.

### 3.3 Escopo do epsilon e do DP-SGD

No comando direto:

- `update_config_from_args` copia `--epsilon` para `config.epsilon` e habilita DP por padrao, salvo se `--no_dp` for passado.
- A geracao/agregacao de embeddings nao consulta `config.epsilon`; usa `teacher_aggregation_noise`.
- `train_model_with_dpsgd` passa `config.epsilon` ao `PrivacyEngine.make_private_with_epsilon` como `target_epsilon`.
- O otimizador recebe `model.parameters()` sem congelar a BERT. Logo, o DP-SGD ajusta a tabela de embeddings, o encoder e a cabeca classificadora, nao apenas a cabeca.
- Os embeddings exportados pelo `save_model_embeddings` sao extraidos apos o treinamento final. Portanto, os vetores exportados refletem tambem as atualizacoes do DP-SGD, embora os alvos de embedding dos teachers tenham sido gerados antes.

O epsilon contabilizado nesse fluxo e o da etapa final DP-SGD. Se a etapa anterior de teachers/agregacao tivesse uma garantia DP propria, seu custo ainda precisaria ser composto com o custo do DP-SGD para declarar uma garantia total. Como essa primeira garantia nao esta demonstrada no codigo, o log nao sustenta uma afirmacao de que todo o pipeline tenha garantia `(epsilon=2.0, delta=1e-5)`.

## 4. Conclusoes desta observacao

1. **Ordem:** os alvos de embedding agregados dos teachers sao gerados e destilados antes do DP-SGD.
2. **Papel do DP-SGD:** ele faz o fine-tuning final e modifica todos os parametros treinaveis, incluindo embeddings; nao calibra o ruido usado anteriormente nos alvos dos teachers.
3. **Privacidade da destilacao:** teachers disjuntos mais ruido podem compor um mecanismo DP, mas nao constituem automaticamente uma garantia. O codigo atual nao demonstra clipping/sensibilidade, calibracao de ruido nem composicao para essa etapa.
4. **Escopo do epsilon observado:** para o comando apresentado, `--epsilon 2.0` e o alvo do DP-SGD final, nao um budget end-to-end demonstrado para a destilacao inteira.
5. **Qualidade dos embeddings:** o colapso para previsoes positivas indica baixa utilidade da classificacao final nesse experimento, mas nao identifica isoladamente a qualidade dos embeddings como causa.
6. **Avaliacao:** validacao e teste usam os mesmos 872 exemplos nesse fluxo; e necessario um conjunto independente para estimar generalizacao sem selecao sobre os mesmos dados.

## 5. Recomendacoes para fechar a analise

### Privacidade e fidelidade metodologica

1. Definir explicitamente a relacao de vizinhanca (por exemplo, um registro de treino) e quais artefatos/saidas sao protegidos.
2. Usar uma fonte de consultas realmente publica e independente, ou incluir a selecao das consultas derivadas do treino na analise de privacidade.
3. Implementar e documentar um limite de sensibilidade para as representacoes agregadas, incluindo o clipping utilizado.
4. Calibrar o ruido de embedding e qualquer mecanismo de consenso ao epsilon/delta da etapa, e contabilizar/compor as consultas com um accountant apropriado.
5. Analisar a contagem de frequencia e a ramificacao `token_freq == 0`; nao tratar dados auxiliares derivados do treino como gratuitos em termos de privacidade.
6. Compor a privacidade da destilacao com a do fine-tuning DP-SGD. Nao rotular o epsilon do ultimo estagio como epsilon global sem essa composicao.
7. Alinhar a implementacao e a descricao: tipos de representacao transferidos, uso de saidas/rotulos agregados, rare-word mechanism, noise injection e perdas do Student.

### Utilidade e experimentacao

1. Registrar distribuicao das previsoes, matriz de confusao e distribuicao de logits para cada epsilon.
2. Avaliar os embeddings antes do DP-SGD com uma sondagem simples e congelada, comparando: embedding original, embedding destilado sem ruido e embedding destilado com ruido.
3. Executar um baseline nao privado e um baseline DP-SGD sem destilacao com o mesmo split, seed, amostras e arquitetura.
4. Separar avaliacao e teste; selecionar checkpoints apenas na validacao e reportar o resultado final em um conjunto de teste independente.
5. Repetir seeds ou execucoes suficientes para separar variacao aleatoria de um efeito consistente de epsilon.

## 6. Limites e rastreabilidade

Esta observacao e uma analise estatica do documento-base e do codigo do repositorio, combinada com o log SST-2 fornecido no notebook. Ela nao e uma prova formal de privacidade, uma auditoria matematica completa do teorema do documento-base nem uma reproducao nova do experimento. Antes de publicar uma garantia formal, e necessaria uma analise do mecanismo implementado, das hipoteses de vizinhanca e da composicao de todos os estagios.

Referencias principais:

- Documento-base: [`dped_multi_layer_noise_distillation_for_privacy_preserving_text_embeddings.md`](../dped_multi_layer_noise_distillation_for_privacy_preserving_text_embeddings.md)
- Fluxo do metodo: [`main.py`](../../main.py#L401)
- Treinamento e agregacao de teachers: [`models.py`](../../models.py#L224)
- Geracao dos embeddings agregados: [`models.py`](../../models.py#L330)
- Destilacao por MSE: [`models.py`](../../models.py#L466)
- Amostragem do conjunto chamado de publico: [`data_loader.py`](../../data_loader.py#L257)
- Configuracao do DP-SGD e epsilon-alvo: [`training.py`](../../training.py#L175)
- Injecao de ruido no `StudentModel.forward`: [`models.py`](../../models.py#L54)

## 7. Atualizacao apos a implementacao da calibracao (2026-11-17)

A observacao acima descreve o codigo anterior a esta mudanca. O fluxo de embeddings foi atualizado para:

- usar todos os IDs fixos do vocabulario do tokenizer, em vez de selecionar tokens a partir do conjunto chamado de publico;
- aplicar clipping L2 por vetor de token nos teachers, com limite definido pelo percentil 95 das normas da tabela pre-treinada publica do Student;
- calibrar o ruido Gaussiano por token com o accountant RDP do Opacus, compondo uma consulta por token para o epsilon e delta informados;
- remover a frequencia de tokens e o fallback baseado nela da geracao e da exportacao DPED;
- exportar os embeddings antes do fine-tuning downstream, com IDs, labels e metadados do mecanismo.

O comando `--embedding_privacy_sweep` executa a grade `epsilon = 1, 2, 4, 8` e nao executa DP-SGD downstream. O delta inicial permanece `1e-5`, mas a execucao valida que `delta < 1/N` para o numero efetivo de exemplos privados. O clipping e calibrado a partir do Student pre-treinado, sem consultar os dados privados.

O preflight com Opacus 1.4.0, `delta=1e-5` e 30.522 tokens produziu multiplicadores RDP de 710, 376,25, 202,5 e 111,40625 para epsilon 1, 2, 4 e 8, respectivamente. O desvio-padrao aplicado ainda multiplica esses valores pela sensibilidade `2C/K`; portanto, a utilidade pode cair substancialmente. O resultado deve ser avaliado diretamente contra os ataques de inversao, sem reduzir silenciosamente o vocabulario ou o ruido.

Cada artefato da grade tem o epsilon indicado individualmente. A liberacao conjunta de varias tabelas da mesma grade requer compor os mecanismos; a grade nao tem, como conjunto, o epsilon de uma de suas execucoes. Qualquer DP-SGD executado depois e outra etapa e tambem precisa ser composta antes de declarar uma garantia total.

O preflight e os testes unitarios validam a calibracao e as invariantes locais, mas nao constituem uma prova formal nem substituem uma execucao completa com os dados do experimento e a avaliacao dos ataques.
