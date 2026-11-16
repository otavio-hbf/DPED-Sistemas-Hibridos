Integrante: Otávio Henrique Baudel Francisco
Artigo escolhido: DPED: Multi-Layer Noise Distillation for Privacy-Preserving Text Embeddings
Autores: Shuya Feng; Yuan Hong
Veículo: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)
Ano: 2025
Justificativa e relevância
O artigo propõe o DPED, um framework para treinamento de representações textuais sob Differential Privacy, combinando teacher-student distillation, injeção de ruído em múltiplas camadas e tratamento específico de palavras raras. Os experimentos demonstram ganhos de utilidade em diferentes níveis de privacidade e comparam o método a abordagens como DP-SGD e PATE. A escolha do trabalho é relevante para a disciplina por combinar aprendizado profundo, processamento de linguagem natural e privacidade, além de apresentar uma oportunidade de avaliação de segurança que não é explorada experimentalmente pelo artigo: a resistência das representações privadas a ataques de inversão de embeddings.
Direção de evolução
Propõe-se avaliar empiricamente a robustez das representações produzidas pelo DPED contra ataques generativos de inversão de embeddings, utilizando o ALGEN, método de ataque proposto por Chen, Xu e Bjerva (ACL 2025). O ALGEN realiza alinhamento entre o espaço de embeddings da vítima e o espaço do atacante e, posteriormente, reconstrói o texto com um modelo generativo pré-treinado. A evolução investigará se as garantias e mecanismos de privacidade do DPED também se traduzem em menor capacidade de reconstrução textual. Para isso, serão comparadas representações BERT não privadas e representações produzidas pelo DPED em diferentes níveis de ε. A avaliação utilizará SST-2, dataset empregado tanto no DPED quanto no ALGEN, permitindo manter a metodologia alinhada aos trabalhos de origem. A análise terá como foco a relação entre orçamento de privacidade e capacidade de inversão.





Objetivo geral
Avaliar a robustez das representações produzidas pelo DPED contra ataques de inversão de embeddings, utilizando o ALGEN como modelo atacante.
Objetivos específicos
Implementar o pipeline de inversão baseado no ALGEN para atacar as representações produzidas pela implementação BERT do DPED.
Comparar a capacidade de reconstrução entre BERT não privado e DPED sob diferentes orçamentos de privacidade.
Analisar o efeito da quantidade de dados disponíveis ao atacante sobre o sucesso da inversão.
Hipóteses
H1: Representações produzidas pelo DPED apresentarão menor capacidade de reconstrução textual do que representações BERT não privadas.
H2: A capacidade de reconstrução aumenta com o aumento do orçamento de privacidade ε.
H3: A quantidade de embeddings vazados ao atacante afetará o sucesso do ataque, sendo possível caracterizar diferentes níveis de risco de inversão para cada configuração de privacidade.
Metodologia
Será utilizado o dataset SST-2, empregado em ambos artigos. No DPED, o dataset será utilizado para produzir as representações BERT sob as configurações estudadas. A condição não privada será utilizada como baseline, enquanto o DPED será avaliado em diferentes níveis de privacidade, como ε ∈ {1, 2, 4, 8}, mantendo δ fixo.
O ataque será realizado utilizando o framework ALGEN, que utiliza um gerador embedding-to-text e uma transformação linear para alinhar o espaço da vítima ao espaço do atacante. O artigo demonstra que o método pode operar com quantidades reduzidas de dados de alinhamento e utiliza até 1.000 amostras em seus experimentos.
A qualidade das reconstruções será medida pelas mesmas métricas utilizadas no ALGEN: BLEU-1, BLEU-2, ROUGE-1, ROUGE-L e similaridade cosseno. 
A execução será planejada para uma infraestrutura com GPU virtual, em plataformas online para execução de notebooks (Kaggle e Colab) priorizando o uso de modelos pré-treinados e o alinhamento linear do ALGEN, que não requer treinamento iterativo da transformação entre os espaços de embeddings. O escopo será mantido compatível com a execução individual do projeto, concentrando os experimentos principais na comparação entre BERT não privado e DPED.

