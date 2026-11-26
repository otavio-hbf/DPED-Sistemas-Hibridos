# Plano de implementação: melhoria da metodologia do experimento ALGEN × DPED

Projeto: ataque de inversão ALGEN contra tabelas de embeddings estáticos produzidas por um pipeline DPED-like, avaliado em SST-2 com professores BERT-base.

Objetivo: transformar o experimento atual em um teste capaz de responder, com controles e incerteza, três perguntas:

1. O ataque consegue recuperar algum conteúdo acima do nível de chance, mesmo sem defesa?
2. A perturbação altera o sucesso do ataque, e em qual faixa de SNR?
3. As hipóteses H1 (DPED < BERT), H2 (reconstrução cresce com ε) e H3 (cresce com n) se sustentam com estimativas, intervalos e controles?

Princípio de ordenação: nenhuma interpretação de ε, H1 ou H2 deve ser feita antes das fases 0 a 3. Sem proveniência verificada e sem controles de piso, qualquer diferença observada é ambígua.

---

## Fase 0. Auditoria de proveniência e reprodutibilidade

Critério de conclusão: cada número publicado pode ser rastreado até um artefato identificado e regenerado a partir dele.

**Estado:** controles implementados para novas execuções, ainda não validados
no Colab. Os artefatos históricos e um lock resolvido para o runtime efetivo
ainda são necessários. Detalhes e procedimento de inventário em
[auditoria-proveniencia-fase0.md](auditoria-proveniencia-fase0.md).
Os notebooks persistem exports, preflight, smoke e cada resultado da grade no
Drive imediatamente após a conclusão de cada etapa, sob um `RUN_ID` exclusivo.

- [ ] Recuperar o PDF-fonte, os arquivos de resumo e as saídas por sentença da análise histórica.
- [ ] Gerar hash (SHA-256) de cada matriz exportada e registrar, para cada rótulo de ε, o hash da matriz efetivamente carregada pelo script de ataque.
- [ ] Verificar, por hash, que as cinco condições (BERT, ε = 8, 4, 2, 1) carregam matrizes distintas. Se houver coincidência, parar e corrigir antes de continuar.
- [ ] Registrar para cada execução: hash do código, revisão do repositório, checkpoint, semente, configuração completa (C, K, z, σ, α, n, número de sentenças) e versões de dependências (fixar versões em arquivo de lock).
- [ ] Excluir a célula contaminada (ε = 8, n = 10) e qualquer resultado de smoke run do conjunto de dados de pesquisa. Manter smoke tests em pasta separada, com rótulo explícito.
- [x] Seleção de sentenças definida e registrada para novas execuções: primeiras elegíveis da validação; hashes conferidos entre condições. O procedimento histórico continua não verificado.
- [x] Documentar a divergência entre a documentação antiga (seleção de tokens públicos e tratamento de frequência) e o código atual.

---

## Fase 1. Diagnóstico do pipeline de ataque (sem defesa)

Objetivo: descobrir se o ataque tem capacidade de recuperação antes de medir qualquer defesa.

- [ ] Implementar e registrar a contagem de strings distintas produzidas pelo decodificador por condição e por n. Se a contagem for 1 ou 2 em n = 1 e n = 10, o resultado é degenerado.
- [ ] Verificar matematicamente o caso n = 1. Com x normalizado e ridge com α, a predição zW é proporcional a (z·x₁) y₁ᵀ, então após a normalização o vetor predito é ±y₁. Confirmar no código se a implementação segue essa fórmula. Se seguir, a saída em n = 1 é quase constante por construção, e isso deve ser reportado como achado, não como resultado do ataque.
- [ ] Implementar os preditores de piso:
  - preditor constante (sempre o mesmo texto, escolhido pelo melhor ajuste no conjunto de validação de treino, ou a média dos alvos decodificada);
  - preditor genérico por frequência (sentença ou conjunto de tokens mais frequentes do domínio);
  - ROUGE-1 e content-word recall para cada um, em cada n.
- [ ] Reportar, para cada condição e n, a diferença em relação aos preditores de piso. Sem essa diferença, nenhum valor de ROUGE-1 pode ser interpretado como recuperação de informação.
- [ ] Recalcular o cosseno com baseline: cosseno do preditor constante (média dos alvos) contra o alvo, em cada n. Reportar o cosseno do ataque apenas como excesso sobre essa baseline.
- [ ] Avaliar a anisotropia dos alvos: média das similaridades par a par entre embeddings alvo. Se for alta, centralizar os alvos (subtrair a média do conjunto de treino) e repetir a avaliação, reportando as duas versões.

Critério de conclusão: é possível dizer, com números, se o ataque supera o piso em cada n e se a saída varia entre sentenças.

---

## Fase 2. Controle positivo com tabela limpa

Objetivo: separar o gargalo do pipeline (ALGEN e objeto de média estática) do efeito de qualquer defesa.

- [ ] Implementar decodificação direta de conjunto de tokens a partir da média da tabela BERT limpa, sem ALGEN: para cada sentença, recuperar o conjunto de tokens cuja soma/média melhor aproxima o vetor observado (por exemplo, busca gulosa ou por otimização sobre o vocabulário).
- [ ] Medir content-word recall, stopword recall e precisão de conjunto nessa decodificação.
- [ ] Comparar com o ALGEN sobre a mesma tabela limpa, nos mesmos n.
- [ ] Interpretação prevista:
  - se a decodificação direta tiver recall alto e o ALGEN baixo, o gargalo é a etapa de alinhamento/decodificação, e o ataque precisa ser melhorado antes de qualquer conclusão sobre defesa;
  - se ambas tiverem recall baixo, o objeto de ataque (média de vetores estáticos) já limita a recuperação, e o experimento deve ser reformulado como estudo do objeto, como já previsto no relatório.

Critério de conclusão: existe um teto de recuperação medido sem defesa, com o qual as condições DP podem ser comparadas.

---

## Fase 3. Caracterização da relação sinal/ruído

Objetivo: quantificar, antes de rodar o ataque, quanto sinal sobrevive ao mecanismo em cada ε.

- [ ] Implementar o cálculo, para cada condição, de:
  - norma média de linha da tabela BERT e do valor C (0,95 quantil);
  - σ_ε = z_ε · 2C/K, com z_ε, K e C efetivamente usados (lidos do artefato, não da documentação);
  - razão sinal/ruído por linha: SNR = ‖sinal‖ / (σ · √d), com d = 768. Pela aproximação, SNR ≈ K / (2 · z_ε · √d).
- [ ] Medir empiricamente, nas matrizes exportadas, a norma média do ruído por linha (diferença entre a tabela exportada e a tabela não ruidosa de referência, quando disponível) e comparar com o cálculo teórico.
- [ ] Verificar a contabilidade conjunta: multiplicador efetivo z_ε / √V (com V = 30 522). Registrar se os valores são coerentes com a regra aproximada ε ≈ √(2 ln(1,25/δ)) / z_eff, para δ = 10⁻⁵. Isso é uma verificação de ordem de grandeza, não substitui a análise formal da Fase 6.
- [ ] Registrar o número de exemplos por professor (N_treino / K) e discutir se K é realista.

Critério de conclusão: tabela com SNR, σ e número de exemplos por professor para cada ε e para os K testados, anexada ao relatório.

---

## Fase 4. Varreduras de ruído e de K

Objetivo: encontrar a faixa de SNR em que o ataque começa a responder. Esta é a análise que responde à pergunta "a perturbação é insuficiente ou excessiva?".

- [ ] Varredura de σ em escala logarítmica, mantendo todo o resto fixo. Sugestão: fatores de 10⁻³ a 10³ em relação a σ_ε, com pelo menos 13 pontos.
- [ ] Varredura de K em {10, 100, 1000} (quando viável), mantendo ε fixo, já que K altera o SNR linearmente.
- [ ] Para cada ponto, reportar ROUGE-1, content-word recall, cosseno com baseline e distância para os preditores de piso, com intervalo bootstrap.
- [ ] Identificar o SNR de transição: faixa em que a métrica sai do piso. Se a transição estiver fora do intervalo de ε usado, a ausência de efeito de ε nos resultados é explicada pela saturação do mecanismo.
- [ ] Interpretação prevista: se o ataque só responde para SNR ≳ 1, o grid atual está na região saturada. Se responde já para SNR ≪ 1, o problema está na sensibilidade do ataque ao ruído ou no alinhamento.

Critério de conclusão: curva de resposta do ataque em função do SNR, com o piso e a transição identificados.

---

## Fase 5. Redesenho da etapa de alinhamento

Objetivo: garantir que o alinhamento não seja a causa dos resultados degenerados nem subestime o ataque por construção.

- [ ] Varrer α no ridge (por exemplo, {10⁻⁴, 10⁻², 1, 10²}) com seleção em conjunto de validação separado, não no conjunto de teste.
- [ ] Repetir com n ≫ d quando possível, para verificar se o regime de regressão explica a tendência de H3.
- [ ] Comparar com alternativas de alinhamento:
  - mínimos quadrados sem regularização (para n ≥ d);
  - total least squares, que trata o ruído também em X;
  - alinhamento com atacante que conhece σ, estimando e removendo o ruído antes da regressão (ver Wang et al., 2026, sobre denoising-aware inversion).
- [ ] Registrar o efeito de atenuação: norma de W em função de σ. Se W encolher com o ruído, a saída tende ao constante, o que deve entrar como explicação alternativa nas análises.
- [ ] Padronizar a normalização: aplicar a mesma transformação (centralização, normalização) a alvos e entradas em todas as condições, e documentar.

Critério de conclusão: a escolha de α e do estimador é feita em validação separada, e o efeito do regime de regressão sobre a saída é conhecido.

---

## Fase 6. Controles de defesa

Objetivo: isolar o efeito do ruído do efeito da destilação e da própria tabela.

- [ ] Baseline destilado sem ruído: mesmo pipeline de destilação, sem nenhum ruído. Isso separa o efeito de destilar do efeito de ruidar.
- [ ] Tabela aleatória com a mesma distribuição de norma da tabela DPED (controle de piso de tabela).
- [ ] Tabela BERT embaralhada por linha (mesma distribuição de valores, sem correspondência com os tokens).
- [ ] Verificar as condições sob a mesma semente de treino do professor e a mesma partição, para que a única variável seja o mecanismo.

Critério de conclusão: cada diferença atribuída ao ε pode ser comparada com um controle que não tem esse mecanismo.

---

## Fase 7. Protocolo estatístico

Objetivo: transformar comparações pontuais em estimativas com incerteza.

- [ ] Número de sementes: no mínimo 3 para cada condição. Onde possível, variar também a partição dos professores e a realização do ruído, que são fontes de variância separadas.
- [ ] Conjunto de teste fixo entre condições, com identificadores de sentença registrados.
- [ ] Tamanho amostral: aumentar de 200 para pelo menos 1000 sentenças, ou justificar o tamanho com análise de potência para a menor diferença que se deseja detectar.
- [ ] Intervalos de confiança por bootstrap sobre sentenças, com pareamento por identificador de sentença para comparações entre condições. Para comparações entre ε, usar diferenças pareadas.
- [ ] Modelo para H2 e H3: regressão com efeitos mistos (sementes e partições como efeitos aleatórios) ou ANOVA com blocos. Reportar estimativa, intervalo e p-valor corrigido para multiplicidade (por exemplo, Holm).
- [ ] Definir antes de rodar, por escrito, o que conta como evidência a favor de cada hipótese (por exemplo, diferença pareada com intervalo excluindo zero e direção prevista). Registrar essa definição no repositório antes dos experimentos finais.

Critério de conclusão: H1–H3 são reportadas com estimativa, intervalo, controles e critério pré-registrado.

---

## Fase 8. Métricas

- [ ] Reportar, para cada métrica, se a agregação é por sentença ou por corpus, e não misturar as duas.
- [ ] Métricas de token: content-word recall, stopword recall, precisão e recall de conjunto de tokens, e F1 contra o chance control.
- [ ] Métricas de superfície: ROUGE-1, ROUGE-L, BLEU-1 e BLEU-2, sempre acompanhadas dos preditores de piso.
- [ ] Métrica de geometria: cosseno com baseline, e correlação por sentença com a métrica de texto (Spearman), tratada como diagnóstico, não como medida de sucesso.
- [ ] Métrica de identificação, se viável: a proporção de sentenças recuperadas com correspondência acima de um limiar, comparada com a taxa de acerto por acaso do conjunto de candidatos.
- [ ] Verificar se o tratamento de NER (F1 em subconjunto) tem tamanho amostral suficiente para ser reportado.

---

## Fase 9. Exploração de mecanismos alternativos (opcional, separada)

Só iniciar se as fases 0 a 4 mostrarem que o grid atual está saturado ou que o ataque responde, mas o ε não altera o resultado de forma controlada.

- [ ] Projeção para dimensão k < 768 antes do ruído (ver Feyisetan e Kasiviswanathan, 2021). Reduz o termo √d da razão sinal/ruído. Exige nova análise de sensibilidade.
- [ ] Alocação de orçamento por token ou por coordenada, com base em frequência ou importância (ver Meisenbacher e Matthes, 2025; Tsai et al., 2026). Exige contabilidade de composição explícita.
- [ ] Sensibilidade ajustada aos vizinhos locais, em vez do pior caso global 2C/K (ver Bollegala et al., 2023; Margaryan e Gupta, 2026, Ball-DP).
- [ ] Cada variante deve ter sua própria análise de sensibilidade e contabilidade. Não comparar variantes com ε nominal sem recalcular o orçamento efetivo.

Critério de conclusão: uma variante alternativa é comparada com o mecanismo base sob o mesmo protocolo, com orçamento efetivo recalculado.

---

## Fase 10. Contabilidade formal e sensibilidade

Objetivo: separar a garantia formal da evidência empírica.

- [ ] Revisar a sensibilidade: a suposição de que uma substituição altera no máximo uma partição, e a consequência para ℓ₂ = 2C/K.
- [ ] Justificar o uso de C calculado sobre a tabela pré-treinada: dizer explicitamente se é dado público e por que não depende de dados protegidos. Caso contrário, incluir o custo de privacidade da escolha de C.
- [ ] Verificar a composição: os V registros são liberados simultaneamente, então a composição deve ser calculada sobre o mecanismo conjunto. Documentar a equivalência (multiplicador conjunto z/√V) e recalcular os valores de ε com um contador independente.
- [ ] Incluir na composição qualquer liberação posterior, incluindo o treino downstream e a seleção de checkpoint, quando usarem dados privados.
- [ ] Documentar o que a garantia cobre e o que não cobre (inferência sobre a tabela publicada; não cobre o encoder contextual nem a inversão de frases completas).
- [ ] Submeter a análise de sensibilidade e o amostrador gaussiano a revisão numérica independente.

Critério de conclusão: a seção de privacidade do relatório declara o orçamento efetivo de cada liberação e as hipóteses da contabilidade.

---

## Fase 11. Infraestrutura e reprodutibilidade

- [ ] Um único script de pipeline que recebe um arquivo de configuração e produz: matriz exportada, hashes, saídas por sentença, métricas e tabelas do relatório.
- [ ] Todas as tabelas e figuras do relatório geradas por script a partir das saídas brutas. Nenhum número digitado à mão.
- [ ] Arquivo de configuração versionado; cada execução grava uma cópia do arquivo junto com os resultados.
- [ ] Testes automatizados: (a) cada condição carrega matriz distinta; (b) o cálculo de σ bate com a fórmula; (c) o caso n = 1 produz a saída prevista pela álgebra; (d) o amostrador gaussiano tem a variância esperada.
- [ ] Separar diretórios de smoke test e de pesquisa, com regra explícita de que smoke tests não entram em tabelas finais.
- [ ] Registrar tempo, memória e orçamento consumido por execução. Investigar e documentar o erro de falta de memória na GPU durante o backward com gradientes por amostra, já que afeta a viabilidade das varreduras.
- [ ] Fixar versões de dependências em arquivo de lock.

---

## Riscos e decisões em aberto

- **K e C reais:** o cálculo de SNR depende de valores que precisam ser lidos dos artefatos. Não usar valores de documentação.
- **Sentenças de avaliação:** a sobreposição entre treino do decodificador e validação do SST-2 não é excluída pela separação atual. Decidir se é necessário um conjunto de avaliação externo.
- **Domínio estreito:** SST-2 é um domínio curto e com vocabulário restrito. Decidir se o estudo será limitado a esse domínio e declarar a limitação, ou se incluirá um segundo domínio.
- **Escopo:** o objeto do ataque é a tabela estática. Inversão de embeddings contextuais exige protocolo próprio e não deve ser misturada aos resultados desta fase.
- **Tempo de execução:** as varreduras das fases 4 e 6 multiplicam o número de execuções. Estimar o custo antes de iniciar e priorizar os pontos mais informativos (faixa de transição e K).

---

## Ordem recomendada de execução

1. Fase 0 (proveniência) e Fase 3 (SNR), que dependem só de artefatos e cálculos.
2. Fase 1 (diagnóstico e pisos) e Fase 2 (controle positivo).
3. Fase 4 (varreduras), com decisão sobre a faixa de interesse.
4. Fases 5 e 6 (alinhamento e controles).
5. Fase 7 (protocolo estatístico com as execuções finais) e Fase 8 (métricas).
6. Fase 10 (contabilidade) em paralelo desde o início, com revisão antes do relatório final.
7. Fase 9 apenas se justificada pelos resultados das fases anteriores.
8. Fase 11 ao longo de todo o trabalho.
