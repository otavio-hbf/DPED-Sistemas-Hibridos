# Relatório integrado de implementação: DPED e ataque de inversão ALGEN

**Data de consolidação:** 23 de novembro de 2026  
**Escopo:** evidências disponíveis em `docs/`, confrontadas com o código, os resultados e os notebooks presentes no repositório.  
**Natureza:** síntese documental e inspeção estática; não foram executados novos treinos, ataques ou testes para este relatório.

## Resumo executivo

O projeto investiga se representações textuais produzidas por uma adaptação do DPED dificultam a reconstrução de texto por um atacante baseado no ALGEN. O objeto principal não é a classificação de sentimento: é a tabela de embeddings do estudante antes do treinamento downstream e a informação textual que pode ser inferida a partir dela.

A implementação atual constrói uma tabela estática de vetores por token, agregando professores treinados em partições disjuntas, limitando a norma dos vetores e adicionando ruído calibrado. Um estudante é ajustado à tabela ruidosa e seus vetores são exportados com IDs e rótulos lexicais. O ataque, por sua vez, forma uma representação de frase pela média dos vetores dos tokens, aprende um alinhamento linear a partir de pares conhecidos e usa um decodificador ALGEN pré-treinado e congelado. Portanto, o experimento **não mede diretamente a inversão de embeddings contextuais**.

O principal resultado de ataque disponível é indireto: uma análise anterior baseada em um PDF do notebook, sem os CSVs, sumários e metadados originais. Nela, o ROUGE-1 aumenta aproximadamente de **0,09 para 0,25** quando os pares de alinhamento passam de 1 para 1.000, mas as diferenças entre BERT e as condições rotuladas por ε são pequenas e não sustentam as hipóteses de proteção ou de tendência monotônica. Uma execução de teste foi misturada a uma célula da grade e a proveniência das tabelas não foi confirmada. Consequentemente, esses números descrevem a execução analisada; **não comprovam o efeito do mecanismo de calibração atual**.

A evidência de privacidade é igualmente delimitada: o mecanismo atual foi projetado para uma liberação por tabela, com contabilização da composição sobre o vocabulário. A interpretação depende das hipóteses de adjacência, partição, fonte pública dos parâmetros e implementação numérica; os testes e o pré-cálculo documentados não são, por si sós, prova formal. A publicação conjunta de várias tabelas ou de um modelo posteriormente ajustado requer análise de composição. Os logs históricos de DP-SGD não são orçamento global da versão atual do pipeline.

**Prioridade imediata:** recuperar os artefatos originais, comprovar qual tabela foi efetivamente atacada, separar a execução de teste e repetir a avaliação com controles e sementes adicionais. Só depois é justificável discutir o efeito de ε sobre o risco de inversão.

## 1. Problema e contexto científico

Embeddings podem preservar informações sobre o texto que lhes deu origem. Um mecanismo de privacidade diferencial (DP), quando corretamente especificado e implementado, limita a influência de uma unidade protegida sobre a distribuição das saídas; não promete que palavras públicas sejam impossíveis de recuperar nem que qualquer ataque tenha sucesso zero. A avaliação empírica de inversão e a garantia formal respondem, assim, a perguntas relacionadas, mas distintas.

O [projeto de pesquisa](projeto.md) parte da proposta DPED de destilação entre professores e estudante e acrescenta uma avaliação de segurança com ALGEN. O [texto de referência DPED](dped_multi_layer_noise_distillation_for_privacy_preserving_text_embeddings.md) descreve professores em partições disjuntas, agregação de sinais com ruído, destilação em múltiplas camadas e tratamento de palavras raras. Relata resultados para arquiteturas e tarefas próprias, incluindo skip-gram, WikiText-2 e IMDb. Esses resultados pertencem ao documento de referência: **não são resultados reproduzidos por este repositório nem devem ser transferidos automaticamente à adaptação BERT/SST-2**.

O [ALGEN](../../algen/SIH-ALGEN-main/README.md) combina alinhamento entre espaços vetoriais e geração textual. Neste projeto, o gerador é mantido congelado e o componente ajustado pelo atacante é o alinhamento; o retreinamento do decodificador não integra o método estudado.

As hipóteses estabelecidas no projeto são:

- **H1:** representações DPED levam a menor reconstrução do que o BERT não privado.
- **H2:** a capacidade de reconstrução aumenta quando ε aumenta, mantidas as demais condições.
- **H3:** a quantidade de pares conhecidos pelo atacante influencia o sucesso da inversão.

O objetivo é examinar essas hipóteses no cenário especificado, não demonstrar que uma métrica de classificação downstream é equivalente a privacidade.

## 2. Objeto da implementação e evolução metodológica

### 2.1 O que é liberado

A saída estudada é a tabela de *lookup* de palavras do estudante, exportada após a destilação e antes do ajuste supervisionado final. Cada linha corresponde a um ID do vocabulário e a seu token textual; esse token pode ser uma palavra, fragmento WordPiece ou token especial. O mecanismo atual exporta o vocabulário fixo completo, inclusive padding e tokens especiais. O ataque exclui tokens especiais ao construir vetores de frases. A associação entre linha, ID e string é explícita.

Essa tabela não contém um vetor diferente para cada ocorrência de uma palavra em seu contexto. O ataque constrói uma frase pela média das linhas correspondentes aos tokens presentes, seguida de normalização. Esse procedimento perde informação de ordem no vetor de entrada e mede o vazamento de uma representação composta a partir de embeddings estáticos; não equivale à saída contextual do encoder BERT. A exportação posterior ao downstream é outro artefato e seus rótulos por split são classes da tarefa, não rótulos lexicais. Fontes: [exportação](../implementação/exportacao-embeddings-dped.md) e [comparação entre exports](../analises/export-pre-downstream-vs-save-model.md).

### 2.2 Do desenho histórico à liberação calibrada

A [observação histórica de privacidade](observacao_dp_distill_privacidade_embeddings.md) identificou que, na versão então examinada, o ε do log se referia ao DP-SGD final, enquanto o ruído na agregação de embeddings usava outro parâmetro e não vinha acompanhado da mesma demonstração de clipping, calibração e composição. Além disso, decisões baseadas em frequências privadas e o uso de um conjunto chamado público exigiam análise própria. Essa observação é importante como diagnóstico, mas não descreve integralmente o código posterior.

A [atualização de calibração](../implementação/calibracao-ruido-embeddings.md) substituiu a seleção dependente de frequências por um vocabulário fixo completo; passou a limitar cada vetor de professor, calibrar o ruído Gaussiano e compor as consultas de token com um accountant. A função legada de contagem, cuja [otimização](../implementação/otimizacao-contagem-tokens.md) foi documentada anteriormente, permanece para compatibilidade e não determina a geração atual da tabela. Não é correto, portanto, apresentar como comportamento vigente nem a antiga seleção de tokens públicos nem a agregação adaptativa de palavras raras descrita no artigo.

A adaptação atual também não é uma reprodução literal do DPED de referência: os professores usam BERT e a transferência observada na destilação é uma perda entre linhas de embeddings, não a combinação demonstrada no texto-base de representações e votos de saída. A existência de ruído em um caminho de forward do estudante tampouco demonstra, por si só, que esse caminho participe da destilação efetivamente usada. Essa distinção delimita o que pode ser atribuído ao artigo e o que pertence à implementação local.

### 2.3 Método de privacidade proposto

A unidade declarada é um exemplo de treino, com adjacência por substituição. Para cada token, as saídas dos professores são limitadas em norma por um valor derivado da tabela pré-treinada, tratada como referência pública independente dos dados protegidos. Com **K** partições disjuntas e estáveis, a mudança de um registro afeta no máximo um professor; a sensibilidade L2 por vetor médio é então limitada por **2C/K**, com **C** o limite de norma. A composição considera uma consulta para cada ID do vocabulário.

O pré-cálculo documentado para 30.522 tokens e δ = 10⁻⁵ produziu multiplicadores de ruído de **710**, **376,25**, **202,5** e **111,40625** para ε igual a 1, 2, 4 e 8, respectivamente. O desvio-padrão aplicado é o multiplicador vezes 2C/K; os multiplicadores não são, isoladamente, o desvio-padrão dos vetores exportados. Os números indicam possível perda de utilidade, mas não quantificam o afastamento final do estudante: a destilação e seus hiperparâmetros também influenciam o resultado.

A tabela ruidosa é usada para ajustar o estudante sem novas consultas aos dados privados no caminho de destilação inspecionado; sob as hipóteses do mecanismo, essa etapa é pós-processamento. Porém, uma conclusão formal exige verificar o processo completo: estabilidade das partições sob a adjacência declarada, independência da referência e de quaisquer seleções auxiliares, limites e aleatoriedade implementados, composição e tratamento de metadados e demais saídas. Os professores e intermediários não privatizados não devem ser considerados protegidos apenas porque o estudante passa por essa etapa.

Cada ε da grade se refere a uma liberação individual. Se as quatro tabelas forem compartilhadas, seus custos devem ser compostos; o mesmo vale para um downstream privado sobre os mesmos dados e outras saídas dependentes deles. Um downstream não privado sobre dados protegidos não adquire DP apenas por partir de um estudante privatizado. Métricas e seleção de modelos também requerem análise se dependerem de dados privados. O pré-cálculo e os testes locais não substituem revisão formal independente.

### 2.4 Ataque e modelo de ameaça

O atacante recebe acesso à representação construída a partir da tabela e conhece pares de frases e vetores para aprender um mapeamento. Para cada frase, a origem **X** é a média normalizada dos embeddings estáticos de seus tokens; o alvo **Y** é a representação normalizada do encoder associado ao decodificador ALGEN. O alinhamento linear regularizado é ajustado nos pares conhecidos e aplicado às frases de avaliação. A geração usa o checkpoint ALGEN congelado.

O baseline escolhido é a tabela BERT pré-treinada, indexada pelos mesmos IDs. É uma comparação útil com uma representação pública não ajustada, mas **não isola o efeito do ruído**: o DPED também passou por treinamento de professores e destilação. Uma condição destilada sem ruído, com o mesmo desenho e sem reivindicação de DP, seria necessária para separar melhor esses fatores.

Os pares de alinhamento vêm do treino SST-2 e as frases avaliadas, da validação SST-2. A separação por split evita reutilizar as mesmas linhas por construção, embora a análise de duplicatas textuais e possível sobreposição com o treinamento do decodificador ainda seja necessária. O script atual toma as primeiras frases elegíveis da validação; o relatório histórico descreve 200 frases amostradas. Sem os IDs e parâmetros da execução original, não se deve presumir que o script inspecionado produziu exatamente aquela amostra. Fontes: [plano do ataque](../analises/plano-ataque-algen-dped.md) e [implementação do atacante](../implementação/ataque-algen-sentence-level.md).

## 3. Evidências de execução e resultados

### 3.1 Cobertura e qualidade das evidências

| Evidência | Estado e interpretação |
| --- | --- |
| Análise do ataque em SST-2 | [Relatório anterior](Análise%20dos%20resultados%20do%20ataque%20ALGEN%20sobre%20embeddings%20DPED%20(SST-2).md) baseado no PDF de análise; arquivos brutos e metadados não disponíveis para sua verificação independente. |
| Notebooks de ataque e análise | Presentes, mas sem saídas persistidas na cópia local inspecionada. Os documentos de implementação registram validações limitadas e a necessidade de execução real. |
| Liberações DPED calibradas | A documentação descreve o mecanismo e sua validação unitária, mas os exports DPED da grade não estão nos resultados locais inventariados. Não se pode confirmar sua existência ou conteúdo no Drive a partir desta inspeção. |
| Notebook histórico de treinamento | [dped_medium.ipynb](../../notebooks/dped_medium.ipynb) preserva saídas de quatro execuções de classificação; elas não demonstram que os embeddings analisados no ataque vieram do mecanismo calibrado posterior. |
| Resultados locais | [results_summary.txt](../../results/glue/results_summary.txt) registra duas execuções pequenas não privadas, ambas com oito exemplos por limite de split e acurácia 0,75; são verificações de fluxo, não estimativas representativas. O [CSV GLUE](../../results/csv/glue_results.csv) apresenta métricas `N/A` e não sustenta comparação de desempenho. |
| Export local | A [tabela não privada](../../results/embeddings/glue_sst2_non-private_eps8p0/metadata.txt) tem forma 30.522 × 768 e é pós-treinamento; o valor ε no nome ou metadado não a torna privada. Não é o baseline pré-treinado escolhido para o ataque. |

O inventário e as fontes detalhadas estão no Anexo D. Datas, descrições de implementação e resultados históricos são preservados como tais, sem afirmar que foram reproduzidos nesta consolidação.

### 3.2 Ataque de inversão: resultados relatados

A análise anterior relata 36 execuções carregadas: 35 da grade de cinco condições por sete tamanhos de alinhamento e uma execução *smoke* adicional. O ataque usa uma semente e regularização fixa. Em geral são 200 frases por execução; a *smoke*, com dez frases, contaminou a célula ε = 8, n = 10 e reduziu a interseção para comparações afetadas.

**ROUGE-1 médio por execução, conforme a fonte histórica:**

| Pares conhecidos n | BERT | ε = 8 | ε = 4 | ε = 2 | ε = 1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0,0903 | 0,0903 | 0,0903 | 0,0903 | 0,0903 |
| 10 | 0,1750 | contaminada | 0,1751 | 0,1749 | 0,1751 |
| 30 | 0,1658 | 0,1659 | 0,1636 | 0,1643 | 0,1669 |
| 100 | 0,1809 | 0,1829 | 0,1836 | 0,1843 | 0,1837 |
| 300 | 0,2189 | 0,2199 | 0,2201 | 0,2149 | 0,2220 |
| 500 | 0,2252 | 0,2284 | 0,2251 | 0,2250 | 0,2244 |
| 1.000 | 0,2590 | 0,2523 | 0,2558 | 0,2571 | 0,2574 |

A tabela apoia uma tendência geral de maior sobreposição lexical com mais pares, mas não uma curva estritamente crescente: há queda entre n = 10 e n = 30. A fonte relata inclinação aproximada de 0,049 de ROUGE-1 por década de n em uma regressão, mas a estimativa não substitui repetições independentes dos pares de alinhamento.

O cosseno de alinhamento relatado aumenta de aproximadamente 0,7085 em n = 1 para 0,917 em n = 1.000. Sua correlação por frase com ROUGE-1 é moderada na análise histórica, Spearman ≈ 0,46: proximidade geométrica não é equivalente à reconstrução correta. Para n = 1.000, a fonte relata recuperação aproximada de 0,07 para palavras de conteúdo e 0,5 para *stopwords*, além de F1 acima do controle de acaso ≈ 0,12 contra F1 bruta ≈ 0,25. Essas medidas são diagnósticos úteis, sujeitos à definição de tokenização e do controle usado.

| Hipótese | Leitura dos resultados disponíveis | Limite |
| --- | --- | --- |
| H1 | Não sustentada na análise histórica: diferenças entre DPED rotulado e BERT são pequenas e variam de sinal. | Não demonstrar equivalência ou ausência de proteção; a identidade e a proveniência das tabelas não foram verificadas. |
| H2 | Não sustentada: não aparece uma relação monotônica confiável entre ε e reconstrução. | Sem confirmação do mecanismo e repetições, não há inferência confiável sobre o efeito causal de ε. |
| H3 | Sustentada como tendência descritiva na execução analisada: o desempenho aumenta entre os extremos de n. | Uma semente, um ridge e conjunto limitado de frases não estabelecem estabilidade ou generalização. |

Os quase empates entre condições justificam investigar, entre outras possibilidades, artefatos históricos, caminhos incorretos, similaridade real após destilação ou baixa sensibilidade da métrica. Nenhuma dessas explicações está demonstrada. Multiplicadores grandes tornam a inspeção importante, mas **não provam** que a tabela final do estudante deva necessariamente estar muito distante do BERT.

O documento de implementação informa referências decodificadas pelo tokenizer do modelo alvo; a análise histórica descreve referências que visualmente se parecem com o texto original. É possível que ambas pareçam semelhantes, mas a normalização efetiva precisa ser verificada nos arquivos e na versão executada. BLEU de corpus é apresentado em escala 0–100; ROUGE e cosseno, em geral, em 0–1. Métricas por frase e por corpus não são intercambiáveis.

### 3.3 Treinamento downstream histórico

As saídas preservadas em [dped_medium.ipynb](../../notebooks/dped_medium.ipynb) registram, para ε-alvo de DP-SGD 8, 4, 2 e 1, acurácia e precisão ≈ 0,5092, recall 1,0 e F1 ≈ 0,6748 ao final. Os accountants registram aproximadamente 7,9907; 3,9939; 1,9987; e 0,9916, respectivamente, com δ = 10⁻⁵. Esses são gastos do estágio DP-SGD conforme os logs, não uma garantia composta de toda a cadeia histórica.

As métricas são compatíveis com predição da classe positiva para todos os exemplos; indicam colapso da **classificação final**, não identificam sua causa nem provam que os embeddings pré-downstream sejam inutilizáveis. A observação de ε = 2 e os detalhes da execução estão em [observação histórica](observacao_dp_distill_privacidade_embeddings.md). No carregamento SST-2, avaliação e teste podem usar a mesma validação; seleção de checkpoint e avaliação final nessa mesma base não constituem teste independente.

### 3.4 Estabilidade operacional e correções documentadas

Três dificuldades de execução foram registradas em momentos distintos:

1. **Gradientes por amostra:** incompatibilidade de dimensões em gradientes de posição do BERT com Opacus. A [correção](../implementação/dp-sgd-opacus-fix.md) documenta IDs posicionais explícitos e execuções reduzidas que chegaram à avaliação e à atualização do accountant.
2. **Lotes vazios:** amostragem Poisson pode gerar lote vazio; o formato dos exemplos inicialmente causava erro de tipos, e o BERT também não aceita o forward vazio. A [análise](../analises/analise-erro-pos-distilacao.md) documenta adaptação de formato e um passo privado sem forward, preservando a chamada ao otimizador e ao accountant. Parte da validação de lote vazio no runtime final ficou pendente na própria fonte.
3. **Memória de GPU:** o [log de erro](../analises/error.txt) registra falha de alocação durante o backward em `grad_sample`, com tentativa de aproximadamente 2,79 GiB e cerca de 1,44 GiB livres. É evidência de limitação operacional naquela execução, não uma medição de desempenho típica nem uma refutação do mecanismo.

As evidências reduzidas comprovam apenas os cenários e ambientes documentados. A validação da configuração completa de treinamento, especialmente no runtime de destino e com os artefatos reais, permanece necessária.

## 4. Síntese crítica e limitações

1. **Proveniência anterior à inferência:** sem hashes, metadados e arquivos por execução, o rótulo ε não comprova qual matriz foi lida nem a versão do mecanismo que a gerou. No script atual, ε é um rótulo passado à saída; não é verificado automaticamente contra os metadados do export.
2. **Contaminação de análise:** a ingestão recursiva pode incorporar *smoke* junto à grade. A célula afetada não deve ser interpretada até a exclusão e o recálculo.
3. **Contrastes insuficientes:** BERT pré-treinado não controla a destilação; falta baseline destilado sem ruído e controles de geração genérica e vetores aleatórios.
4. **Incerteza subestimada:** bootstrap e testes pareados sobre frases na mesma execução não medem a variabilidade da escolha dos pares de ataque, da inicialização, do treinamento dos professores ou da realização do ruído. Uma única semente também impede estimar essa variabilidade.
5. **Frases e possíveis dependências:** separar treino e validação por split não elimina duplicatas entre splits, exposição anterior ao decodificador nem outras sobreposições. O conjunto de validação SST-2 não é uma população ampla de domínios.
6. **Interpretação de métricas:** ROUGE e BLEU capturam sobreposição, não fidelidade semântica ou identificação de registros. Palavras funcionais frequentes, comprimento e saídas genéricas podem elevar escores. Cosseno alto não garante texto correto.
7. **Escopo de DP:** a declaração por tabela depende de pressupostos e análise completa. A recuperação de uma frase não invalida automaticamente DP, assim como um ataque malsucedido não a comprova.
8. **Reprodutibilidade:** a semente de treino não reproduz o ruído gerado com entropia do sistema; repetir estudos significa preservar artefatos e configurações e avaliar distribuições entre realizações independentes, não divulgar material aleatório sensível nem prometer matrizes idênticas.
9. **Divergências documentais:** o README e análises antigas descrevem seleção por tokens públicos, exclusão de padding ou tratamento de frequência que não corresponde à exportação atual de vocabulário completo. O código e a documentação de calibração mais recente devem prevalecer para descrever a versão inspecionada, sem apagar a história.

## 5. Melhorias propostas e critérios de conclusão

### Prioridade 1 — validar os resultados já relatados

- Recuperar o PDF de origem, `summary.json`, CSVs por frase e exports efetivamente usados, incluindo metadados e IDs. Registrar hash dos artefatos, revisão do código, checkpoint e configuração, sem expor dados privados desnecessários.
- Confirmar, por execução, caminho e hash da matriz carregada, alinhamento de linhas/IDs/labels, origem do baseline, dimensões, finitude e correspondência entre ε declarado e metadados. O hash identifica o arquivo; não prova por si que o mecanismo foi executado corretamente.
- Comparar tabelas e vetores de frase entre ε e BERT: distância relativa, distribuição de normas, similaridade por linha e por frase; verificar que diferentes condições não apontam acidentalmente para o mesmo arquivo. Registrar também o efeito da destilação entre alvo ruidoso e tabela final quando isso puder ser feito com segurança.
- Excluir explicitamente a *smoke*, reconstruir a grade e verificar 35 condições únicas, cobertura de frases, IDs, escalas e igualdade entre agregações e fontes. Recalcular todas as comparações afetadas; não substituir os números atuais por suposições.
- Revisar a origem exata do texto de referência, truncamento e normalização; conferir amostras e recomputar métricas a partir de predições e referências preservadas.

**Concluído quando:** cada número publicado for rastreável a artefatos identificados, a grade estiver livre de contaminação e os relatórios puderem ser regenerados.

### Prioridade 2 — fortalecer o desenho experimental

- Repetir pelo menos três sementes de seleção dos pares e, conforme recursos, múltiplos conjuntos de professores e realizações independentes de ruído. Separar fontes de variabilidade e manter as mesmas frases de teste entre condições.
- Definir o ridge antecipadamente ou selecioná-lo em validação separada, sem usar o teste; documentar a grade e a regra de seleção. Comparar tamanhos de alinhamento mantendo amostragem e análise coerentes.
- Incluir controles de entrada aleatória e predição genérica, além do baseline BERT e de um estudante destilado sem ruído. O controle sem ruído não deve ser apresentado como uma liberação privada.
- Avaliar reconstrução por frase e token: ROUGE/BLEU, correspondência exata quando apropriada, recuperação de palavras de conteúdo, *stopwords*, comprimento, diversidade, repetições e excesso sobre acaso. Explicitar definições e agregações.
- Usar comparações pareadas por ID e intervalos que contemplem repetições independentes; tratar multiplicidade e relatar tamanho de efeito. Aumentar a amostra e verificar duplicatas, domínios e eventual exposição prévia ao decodificador.

**Concluído quando:** H1–H3 forem apresentadas com estimativas, incerteza e controles que permitam distinguir efeito de ε, destilação, n e variação experimental.

### Prioridade 3 — garantia e utilidade

- Submeter a sensibilidade, a partição, a composição por token e a implementação do gerador Gaussiano a revisão formal e numérica independente; registrar explicitamente o conjunto de saídas cobertas.
- Calcular e reportar a composição efetiva das liberações realmente compartilhadas, inclusive downstream e seleção de modelo quando aplicável. Não ajustar ruído, vocabulário ou clipping com base em dados privados sem novo mecanismo e contabilização.
- Quantificar utilidade dos embeddings antes de downstream: similaridade, vizinhança, normas, recuperação lexical e, se necessária, sondagem com protocolo e dados separados. Comparar resultados de privacidade e utilidade sem tratar classificação downstream como objetivo principal.
- Se o custo for alto, avaliar alternativas de mecanismo, granularidade e eficiência somente com nova análise de sensibilidade, contabilização e benchmarks; não reduzir silenciosamente o ruído para obter melhor resultado.

### Prioridade 4 — robustez operacional e extensões

- Reexecutar testes de lotes vazios e fluxo completo no ambiente final, registrar versões, memória, tempos, epsilon gasto e warnings. Investigar o OOM com perfil de memória e tamanho físico de lote, preservando a semântica do lote lógico e do accountant.
- Melhorar a rastreabilidade automatizada entre exports, execução e análise, mantendo *smoke tests* fora dos dados de pesquisa.
- Como extensão separada, avaliar embeddings contextuais do estudante ou inversão token a token. Essas variantes mudam o objeto do ataque e exigem protocolo próprio; não devem ser misturadas aos resultados de médias estáticas.

## 6. Conclusão

O repositório dispõe de um caminho implementado para produzir e exportar embeddings estáticos do estudante e de uma adaptação ALGEN cujo ataque se concentra no alinhamento, sem retreinar o decodificador. A evolução da implementação responde a limitações importantes da versão histórica, especialmente seleção dependente de dados e ausência de calibração específica para a tabela.

Os resultados de inversão disponíveis sugerem maior sucesso com mais pares conhecidos, mas ainda não demonstram que variar ε no mecanismo atual modifica o risco. As evidências são insuficientes para concluir que DPED protege mais, protege menos ou equivale ao BERT nesse ataque. A etapa seguinte é fechar a cadeia de proveniência e executar comparações controladas, mantendo separadas a análise formal de DP, a inversão empírica e a utilidade das representações.

---

# Anexos técnicos

## Anexo A — Fluxo, código e artefatos

| Etapa | Implementação e comportamento relevante |
| --- | --- |
| Configuração e dados | [config.py](../../config.py), [data_loader.py](../../data_loader.py) e [utils.py](../../utils.py) configuram o experimento, carregam/tokenizam dados e particionam o conjunto de treino entre professores. Para SST-2, o carregador usa validação também como teste rotulado; isso não cria um teste independente. |
| Professores | [models.py](../../models.py) cria professores BERT, os treina por partição e extrai suas tabelas de palavras. O clipping global do gradiente no treino de professores não é o clipping de sensibilidade dos vetores agregados. |
| Mecanismo por tabela | [privacy.py](../../privacy.py) valida ε e δ, calcula C pelo quantil 0,95 das normas pré-treinadas, limita linhas e usa `get_noise_multiplier` com accountant RDP, `sample_rate=1` e um passo por token. A entrada de aleatoriedade usa `secrets` e a transformação normal implementada no arquivo; segurança numérica e garantia exigem análise própria. |
| Destilação | [models.py](../../models.py) usa `AdamW` e MSE entre as linhas do estudante e os alvos privatizados. O sweep reutiliza o ensemble, inicializa um estudante por ε e não executa DP-SGD downstream. |
| Orquestração | [main.py](../../main.py) implementa `--embedding_privacy_sweep`, `--skip_downstream`, export pré-downstream e o caminho de treinamento downstream. A grade fixa é ε ∈ {1, 2, 4, 8}. |
| Export | `token_embeddings.npy` e `.pt` contêm a matriz; `token_ids.npy` e `token_labels.json`, o alinhamento lexical; `metadata.txt`, a etapa e os parâmetros de privacidade. O export pós-run usa os rótulos da tarefa em outros arquivos. |
| DP-SGD | [training.py](../../training.py) usa `PrivacyEngine`, accountant RDP e `BatchMemoryManager`; adapta exemplos para o carregador privado e trata lote vazio com `grad_sample` vazio antes do passo do otimizador. O tamanho físico e a semântica do accountant precisam ser verificados no runtime usado. |
| Ataque | [attacker_dped_sentence.py](../../algen/SIH-ALGEN-main/src/attacker_dped_sentence.py) carrega a tabela ou BERT pré-treinado, faz média e normalização, ajusta ridge e usa os componentes ALGEN. `--epsilon` identifica a condição de saída; não valida a origem da matriz. |
| Saídas do ataque | `sentence_results.csv` contém condição, n, ridge, semente, ID, predição, referência, cosseno e ROUGE por frase; `summary.json` contém métricas agregadas e cobertura. BLEU não é coluna por frase no CSV atual. |
| Análise | [analise_resultados_ataque.ipynb](../../notebooks/analise_resultados_ataque.ipynb) lê saídas recursivamente e calcula comparações e diagnósticos. O desenho da análise e suas limitações estão em [notebook-analise-resultados-ataque.md](../implementação/notebook-analise-resultados-ataque.md). |

### Dependências e execução

O [requirements.txt da raiz](../../requirements.txt) declara PyTorch, Transformers `>=4.0,<5`, Datasets `>=2.18,<3`, Opacus `==1.4.0`, NumPy, scikit-learn, tqdm, matplotlib e Hugging Face Hub `<1.0`. O [requirements do ALGEN](../../algen/SIH-ALGEN-main/requirements.txt) inclui, entre outros, pandas, evaluate, sentencepiece, POT e rouge-score; os ambientes devem ser resolvidos e suas versões efetivas registradas, pois nem todas estão fixadas. O README da raiz contém recomendações de versões que não equivalem necessariamente ao ambiente histórico ou ao manifesto atual.

Exemplos de entrada, a executar somente após configurar dados, dependências e checkpoint confiável:

```powershell
# Grade de liberações de embeddings; não executa downstream.
python main.py --embedding_privacy_sweep --dataset glue --task sst2 --delta 1e-5

# Execução única: para depois do export pré-downstream.
python main.py --dataset glue --task sst2 --method dp-distill --epsilon 8 --skip_downstream
```

O [notebook de ataque](../../notebooks/attack_algen_dped.ipynb) e a [variante de calibração](../../notebooks/attack_algen_dped_calibracao_ruido.ipynb) documentam a sequência Colab, *smoke test*, grade e cópia de resultados. Caminhos de Drive, branch e nomes de checkpoint registrados são históricos e precisam ser conferidos antes de execução. Carregamento de checkpoint com desserialização irrestrita só é apropriado para arquivo de origem confiável.

## Anexo B — Parâmetros, critérios e validações documentadas

| Item | Valor ou estado |
| --- | --- |
| Unidade e adjacência declaradas | Um exemplo de treino; substituição. A sensibilidade 2C/K pressupõe que a mudança afeta no máximo um professor. |
| Vocabulário e dimensão de referência | BERT-base: 30.522 IDs e 768 dimensões; vocabulário fixo completo, com exclusão de especiais apenas na construção de X pelo atacante. |
| Calibração documentada | δ = 10⁻⁵; exige δ < 1/N. Multiplicadores 710, 376,25, 202,5 e 111,40625 para ε = 1, 2, 4 e 8, nessa ordem. |
| Ataque | n ∈ {1, 10, 30, 100, 300, 500, 1.000}; a análise histórica registra ridge 1,0, semente 0 e 200 frases, exceto *smoke*. |
| Diagnóstico da grade | 35 condições previstas mais uma *smoke* carregada indevidamente. Não há arquivos brutos locais suficientes para recalcular. |
| Testes históricos | A documentação de exportação relata 16 testes aprovados em seu ambiente; outros testes e alterações posteriores têm status próprio. Não afirmar que a suíte atual foi executada neste relatório. |
| Testes presentes | [tests/](../../tests/) inclui verificações de seed, export, clipping e accountant, lotes Opacus e contagem legada de tokens. Teste unitário não comprova execução representativa nem garantia formal ponta a ponta. |

## Anexo C — Cronologia e rastreabilidade de decisões

| Data ou período | Registro | Situação nesta consolidação |
| --- | --- | --- |
| 05/11/2026 | Correção de dimensões de gradientes Opacus e testes reduzidos; resultados não privados locais com oito exemplos por limite de split. | Evidência histórica de execução e correção específica; não é benchmark representativo. |
| 13–14/11/2026 | Diagnóstico de lotes vazios e otimização da contagem; log posterior de OOM em GPU. | Tratamento de lote vazio e função legada presentes; validação completa no ambiente final não demonstrada pelas fontes. |
| 15/11/2026 | Análise do ε do DP-SGD e das limitações da agregação histórica. | Diagnóstico anterior à calibração atual; não generalizar suas afirmações sobre o código antigo. |
| 16–17/11/2026 | Grade histórica de ataque e análise baseada em PDF; desenho de export pré-downstream e calibração. | Resultados de ataque não vinculados aos exports calibrados por evidência disponível; a versão atual usa vocabulário completo. |
| 23/11/2026 | Consolidação documental e leitura estática desta versão. | Sem novos treinos, testes ou reprodução de ataque. |

## Anexo D — Inventário integral das evidências de `docs/`

| Fonte | Contribuição para o relatório |
| --- | --- |
| [analise-erro-pos-distilacao.md](../analises/analise-erro-pos-distilacao.md) | Diagnóstico do carregador Opacus, lotes vazios, testes reduzidos e pendências de ambiente. |
| [error.txt](../analises/error.txt) | Traceback de OOM de GPU durante backward com gradientes por amostra. |
| [export-pre-downstream-vs-save-model.md](../analises/export-pre-downstream-vs-save-model.md) | Diferença entre snapshot pré-downstream e export final; partes sobre seleção por conjunto público são históricas. |
| [plano-ataque-algen-dped.md](../analises/plano-ataque-algen-dped.md) | Modelo de ameaça, baseline, hipóteses operacionais, grade e controles planejados. |
| [ataque-algen-sentence-level.md](../implementação/ataque-algen-sentence-level.md) | Implementação do atacante, referências, saídas e limitações. |
| [calibracao-ruido-embeddings.md](../implementação/calibracao-ruido-embeddings.md) | Mecanismo, sensibilidade assumida, accountant, multiplicadores e escopo das liberações. |
| [dp-sgd-opacus-fix.md](../implementação/dp-sgd-opacus-fix.md) | Correção histórica de gradientes por amostra, testes e instruções de execução. |
| [exportacao-embeddings-dped.md](../implementação/exportacao-embeddings-dped.md) | Formato, semântica lexical, etapa e teste do export atual. |
| [notebook-analise-resultados-ataque.md](../implementação/notebook-analise-resultados-ataque.md) | Conteúdo analítico do notebook e limites de sementes e validação. |
| [notebook-attack-algen-dped.md](../implementação/notebook-attack-algen-dped.md) | Orquestração Colab, dependências adicionais, smoke e grade; alguns caminhos e afirmações são históricos. |
| [otimizacao-contagem-tokens.md](../implementação/otimizacao-contagem-tokens.md) | Evolução da contagem legada; expressamente superada na liberação atual. |
| [Análise dos resultados do ataque ALGEN sobre embeddings DPED (SST-2).md](Análise%20dos%20resultados%20do%20ataque%20ALGEN%20sobre%20embeddings%20DPED%20(SST-2).md) | Resultados numéricos históricos, contaminação, diagnósticos e limites dos dados disponíveis. |
| [dped_multi_layer_noise_distillation_for_privacy_preserving_text_embeddings.md](dped_multi_layer_noise_distillation_for_privacy_preserving_text_embeddings.md) | Texto de referência DPED e resultados atribuídos aos autores, não reproduções locais. |
| [observacao_dp_distill_privacidade_embeddings.md](observacao_dp_distill_privacidade_embeddings.md) | Diagnóstico histórico do escopo de ε, colapso de classificação e atualização posterior. |
| [projeto.md](projeto.md) | Motivação, objetivos, hipóteses, SST-2 e metodologia proposta. |

As fontes externas citadas nos documentos de referência não foram verificadas independentemente nesta consolidação. Divergências entre relatos históricos, código atual e artefatos estão explicitadas no corpo do relatório.
