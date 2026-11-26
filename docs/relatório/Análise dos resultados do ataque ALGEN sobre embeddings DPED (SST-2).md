# Análise dos resultados do ataque ALGEN sobre embeddings DPED (SST-2)

Nov 17, 2026 · @otavio

## Resumo executivo

O fator dominante é o número de pares vazados (n), não o orçamento ε. O ROUGE-1 sobe de 0,09 com n=1 para 0,25 com n=1000.

- **H1 (DPED < BERT):** não sustentada. As diferenças chegam a 0,007 em ROUGE-1 (2,6%), com sinal inconsistente.
- **H2 (sucesso cresce com ε):** não sustentada. O efeito de ε é indistinguível de zero.
- **H3 (sucesso depende de n):** sustentada na tendência geral, com a ressalva de uma única semente.
- **Problema de validade:** a execução `smoke` está misturada à célula ε=8, n=10. As tabelas de ε quase não diferem do BERT, o que precisa ser verificado antes de qualquer conclusão sobre privacidade.

## Validação e cobertura dos dados

A análise usa só o PDF exportado do notebook. Os arquivos `summary.json`, os CSVs e os metadados das exportações não foram recebidos.

- **Execuções:** 36 carregadas. São 35 da grade (4 valores de ε + BERT, × 7 valores de n) e uma `smoke` com 10 frases, em ε=8, n=10.
- **Contaminação:** a `smoke` entra nas médias da célula ε=8, n=10. A interseção de frases cai para 10, quando deveria ser 200. Excluir a execução e recalcular.
- **Referência:** o exemplo impresso é o texto original do SST-2, minúsculo e tokenizado por espaços. Não é o texto decodificado do flan-t5. Confirmar se BLEU e ROUGE normalizam os tokens da mesma forma.
- **Escalas e coerência:** BLEU em 0–100; ROUGE, EM e cosseno em 0–1. Sumário e CSV coincidem (diferença de 0,0000). A métrica própria e o ROUGE-1 do CSV têm Spearman de 0,991.
- **Divisões:** pool de treino de 67.349 pares (treino do SST-2). Teste de 200 frases amostradas de 872 (validação). Não há sobreposição por construção.
- **Limites:** uma semente (0) e um único ridge (λ = 1,0).
- **Não exibido:** as tabelas de exemplos e de frases fáceis/difíceis aparecem como objetos vazios no PDF.

## Resultados quantitativos

O ROUGE-1 cresce com n em todas as condições e quase não muda com ε. A célula ε=8, n=10 está contaminada pela execução `smoke`.

**ROUGE-1 médio por execução (200 frases)**

| n | BERT | ε=8 | ε=4 | ε=2 | ε=1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0,0903 | 0,0903 | 0,0903 | 0,0903 | 0,0903 |
| 10 | 0,1750 | contaminada | 0,1751 | 0,1749 | 0,1751 |
| 30 | 0,1658 | 0,1659 | 0,1636 | 0,1643 | 0,1669 |
| 100 | 0,1809 | 0,1829 | 0,1836 | 0,1843 | 0,1837 |
| 300 | 0,2189 | 0,2199 | 0,2201 | 0,2149 | 0,2220 |
| 500 | 0,2252 | 0,2284 | 0,2251 | 0,2250 | 0,2244 |
| 1000 | 0,2590 | 0,2523 | 0,2558 | 0,2571 | 0,2574 |

**Cosseno (X·T vs Y).** Vai de 0,7085 em n=1 a 0,917 em n=1000. Em n=1000, as cinco condições diferem em menos de 0,00005.

**Tendência.** A regressão de ROUGE-1 em log₁₀(n) dá inclinação de 0,049 por década, com R² = 0,93. Entre 10 e 30 há uma queda não significativa. Entre 500 e 1000 o ganho é significativo, de +0,024 a +0,034.

**Pares necessários para atingir um limiar de ROUGE-1 (interpolação)**

| Limiar | ε=8 | ε=4 | ε=2 | ε=1 | BERT |
| --- | --- | --- | --- | --- | --- |
| 0,188 | 116 | 114 | 114 | 113 | 123 |
| 0,232 | 555 | 584 | 582 | 587 | 575 |
| 0,254 | não atingido | 960 | 935 | 931 | 903 |

Em 0,188, as condições DPED exigem menos pares que o BERT. A diferença é pequena e não tem intervalo de confiança.

## Análise das hipóteses

H1 e H2 não são sustentadas. H3 é sustentada na tendência geral. Os três vereditos dependem da validação do artefato DPED, discutida nos diagnósticos.

| Hipótese | Veredito | Evidência principal |
| --- | --- | --- |
| H1: DPED reconstrói menos que o BERT | Não sustentada | Δ médio de −0,000. DPED supera o BERT em 43% das configurações. Maior diferença: −0,007 em ε=8, n=1000. |
| H2: a reconstrução cresce com ε | Não sustentada | Em n=1000 a ordem é ε=8 < ε=4 < ε=2 ≈ ε=1, oposta à prevista. Coeficiente de log₂(ε) = −0,0006 (EP 0,0023). |
| H3: o sucesso depende de n | Sustentada (tendência) | ROUGE-1 vai de 0,09 a 0,25. n explica 24% da variância de ROUGE-1 e 79% da do cosseno. |

- **H1:** as diferenças pareadas são pequenas e mudam de sinal entre valores de n. Em n=300, ε=8 e ε=4 ficam acima do BERT. A única diferença negativa com significância está em n=1000, em ε=8, que deveria ser a condição de menor privacidade.
- **H2:** nenhuma diferença entre ε vizinhos é significativa após correção de Bonferroni. O ε explica 0,0% da variância de ROUGE-1.
- **H3:** a queda entre n=10 e n=30 pode ser um regime sub-determinado, já que n < 768 com λ = 1. Isso precisa ser testado com uma grade de ridge e várias sementes.

## Diagnósticos e exemplos

O diagnóstico mais importante é a validade das condições de ε. Os resultados quase não variam com ε, e o cosseno é igual entre condições até a quinta casa decimal.

Três explicações são compatíveis com os dados, e o material recebido não permite descartar nenhuma:

1. A execução usou um export anterior à calibração, e os rótulos de ε não correspondem ao mecanismo atual.
2. O ruído calibrado foi aplicado, mas o ataque usou outra tabela que não a exportada.
3. O ruído é pequeno demais para afetar o alinhamento, e a garantia formal não se traduz em proteção prática.

O documento de calibração prevê multiplicadores de 111 a 710 sobre a sensibilidade. Com ruído desse porte, as tabelas deveriam se afastar muito da BERT. Verificar: metadados e hash de cada export; a razão ‖E\_ε − E\_BERT‖ / ‖E\_BERT‖ por token; a diferença entre os X de cada frase.

- **Colapso de modo:** com n=1, só 10% das predições são distintas, e o token "the" ocupa 27,5% das palavras previstas. Com n ≥ 300, quase todas são distintas.
- **Comprimento:** predições de ≈25 palavras em n=1 e de ≈19–21 em n ≥ 10. A média da referência é 15,4.
- **Nível de palavra:** em n=1000, a recuperação de palavras de conteúdo é ≈0,07, e a de stopwords é ≈0,5. As palavras "alucinadas" mais frequentes são funcionais ou genéricas ("a", "but", "film", "movie").
- **Acima do acaso:** em n=1000, a F1 acima do acaso é ≈0,12, contra F1 bruta de 0,25. Em n=1 ela é zero.
- **Cosseno:** vai de 0,70 a 0,92, com Spearman de 0,46 frente ao ROUGE-1 entre frases. É um indicador fraco de qualidade textual.

**Exemplo representativo (n=1000).** Referência: "the acting , costumes , music , cinematography and sound are all astounding given the production 's austere locales ."

| Condição | ROUGE-1 | Predição (trecho) |
| --- | --- | --- |
| ε=8 | 0,171 | "The film was filmed in the same manner as the film, but it's a remembrance of a plight" |
| ε=4 | 0,216 | "But the film was filmed in the same manner as the film, and it's a plethora of dramas…" |
| ε=2 | 0,167 | "…but it's a reinterpretation of a dazzling performance" |
| ε=1 | 0,229 | "…but it's a reinterpretation of humour and" |
| BERT | 0,216 | "But the film was filmed in the same manner as the film, and it's a plethora of dramas…" |

A predição recupera só "film". Nenhuma palavra de conteúdo da referência aparece. Para a mesma frase, o ROUGE-1 varia de 0,17 a 0,23 entre condições, embora as médias sejam quase iguais.

## Limitações e ameaças à validade

1. **Validade do pipeline.** É a limitação mais grave. Enquanto a proveniência das tabelas DPED não for confirmada, os rótulos de ε não podem ser atribuídos ao mecanismo descrito.
2. **Contaminação pela `smoke`.** Afeta a célula ε=8, n=10 e a interseção de frases.
3. **Uma semente e λ fixo.** A variabilidade da amostragem dos pares e a sensibilidade ao ridge não foram estimadas. Não é possível dizer se a queda entre n=10 e n=30 é real.
4. **Amostra pequena.** Com 200 frases, os intervalos são largos para diferenças da ordem de 0,005.
5. **Alvo estático.** O ataque mede a inversão de médias de vetores estáticos por token. Não mede a saída contextual do encoder DPED.
6. **Garantia por artefato.** Cada ε descreve a liberação de uma tabela. Se a grade inteira for liberada, a garantia conjunta é composta e pior que a de cada ε individual.
7. **Métricas e referências.** ROUGE e BLEU favorecem predições de comprimento fixo. Os controles de acaso usam predições de outras frases da mesma execução. Não há um baseline fixo, como uma frase genérica para todas as entradas.
8. **Limiares.** A tabela de pares necessários depende da interpolação e de limiares escolhidos a partir dos próprios dados.

## Conclusão e próximos passos

Com o pipeline atual, o ataque reconstrói palavras de conteúdo quase só quando o atacante tem algumas centenas de pares. O número de pares vazados é o principal determinante do sucesso. Não há diferença atribuível à privacidade entre as configurações DPED e o BERT. Essa conclusão depende de confirmar que as tabelas liberadas refletem o mecanismo calibrado. Sem essa confirmação, o resultado descreve o artefato analisado, não a garantia de privacidade.

&#32;
