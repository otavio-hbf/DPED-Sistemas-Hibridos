# Plano de otimizacao da contagem de tokens

Data: 2026-11-13
Escopo: contagem de tokens usada na distilacao de embeddings do metodo `dp-distill`

> **Historico/superado:** desde 2026-11-17, a liberacao DPED usa o vocabulario fixo completo e nao consulta frequencias de tokens. `get_token_frequency()` e mantida apenas para compatibilidade com analises e testes legados; nao participa da geracao nem da exportacao DPED atual. Este documento registra a otimizacao do fluxo anterior e nao descreve o mecanismo vigente.

## Objetivo

Reduzir o tempo de execucao e o uso desnecessario de GPU durante a etapa de contagem, sem pular a etapa nem alterar o criterio metodologico usado para identificar tokens raros.

## Situacao atual

Em `models.py`, `TeacherEnsemble.generate_embeddings()` chama `get_token_frequency()` antes de gerar os embeddings agregados. A implementacao atual:

1. Percorre o conjunto publico em lotes.
2. Para cada token unico de cada lote, cria um `DataLoader` para cada conjunto de professor e varre esse conjunto ate encontrar o token.
3. Move os IDs do conjunto publico para o dispositivo de treino e calcula os tokens unicos ali.
4. Carrega um `BertTokenizer` que nao e usado nessa funcao.

Assim, os mesmos conjuntos dos professores podem ser percorridos repetidamente para tokens de lotes publicos diferentes. Essa etapa nao exige inferencia nem operacoes de modelo na GPU.

## Criterio metodologico a preservar

Apesar do nome `token frequency`, o valor atual nao e a frequencia de ocorrencias do token. Ele representa em quantos conjuntos de professores o token aparece pelo menos uma vez. Com o limite padrao de raridade igual a 2, um token e considerado raro quando aparece em menos de dois conjuntos.

A otimizacao deve manter esse significado e o uso desses valores em `generate_embeddings()`. Na validacao foi identificado que a implementacao anterior podia incrementar o mesmo token mais de uma vez por professor quando ele aparecia em lotes publicos diferentes. Essa duplicacao nao representa a presenca por conjunto de professor; a contagem otimizada corrige esse caso e pode, portanto, alterar tais valores numericos. Mudar para contagem total de ocorrencias ou desativar a etapa seria uma alteracao metodologica e fica fora deste plano.

## Plano de execucao

### 1. Registrar uma linha de base

- Medir separadamente o tempo de `get_token_frequency()` em uma execucao representativa.
- Registrar tamanho do conjunto publico, quantidade de professores, dispositivo e tempo total de execucao.
- Guardar os valores de contagem para um conjunto pequeno e deterministico, para comparacao com a implementacao otimizada.

### 2. Calcular presenca uma vez por conjunto de professor

- Percorrer cada `teacher_dataset` uma unica vez.
- Acumular os IDs de tokens presentes em cada conjunto, sem contar repeticoes dentro do mesmo conjunto.
- Somar uma unidade por conjunto de professor em que cada token aparece.
- Restringir o resultado aos tokens do conjunto publico, se isso mantiver a interface atual.

Isso substitui as buscas repetidas por uma passagem pelos dados de cada professor, seguida de consultas em estruturas de presenca.

### 3. Manter a contagem fora da GPU

- Processar `input_ids` na CPU durante essa etapa.
- Evitar transferencias CPU-GPU e sincronizacoes para obter tokens unicos.
- Remover a inicializacao do tokenizer que nao e utilizado pela funcao.

### 4. Validar equivalencia e desempenho

- Comparar a contagem antiga e a nova token a token em um conjunto pequeno e reproduzivel; verificar a igualdade quando nao ha repeticao entre lotes publicos e confirmar que repeticoes entre lotes deixam de inflar a presenca por professor.
- Executar o fluxo `dp-distill` com subconjuntos pequenos e confirmar que a distilacao e o fine-tuning continuam funcionando.
- Medir novamente a etapa isoladamente e a execucao completa, no mesmo ambiente e com os mesmos dados e parametros.
- Registrar o ganho observado e confirmar que a etapa de contagem nao realiza operacoes na GPU.

## Cuidados com cache e reprodutibilidade

Os conjuntos de professores sao particionados a partir dos dados de treino. As contagens podem ser reutilizadas dentro do mesmo `TeacherEnsemble`, mas nao devem ser reaproveitadas entre particoes diferentes sem garantir que os conjuntos de cada professor sao os mesmos. A divisao aleatoria dos dados pode alterar a presenca de tokens por professor e, portanto, os valores usados para classificar tokens raros.

## Fora de escopo

- Adicionar uma flag para pular a contagem.
- Alterar o limite de raridade ou a definicao de token raro.
- Trocar presenca por professor por frequencia total de ocorrencias.
- Alterar a agregacao dos embeddings, o ruido ou os demais mecanismos de privacidade.

## Resultado da implementacao

`TeacherEnsemble.get_token_frequency()` agora coleta os tokens publicos uma vez e percorre cada conjunto de professor uma vez, acumulando presenca em conjuntos na CPU. O resultado continua restrito aos tokens publicos, com uma unidade por conjunto de professor; a inicializacao do tokenizer nao utilizado foi removida.

O teste deterministico `tests/test_token_frequency.py` confirma a presenca por professor, repeticoes dentro de um professor sem incremento adicional, e a correcao do sobrecontador entre lotes publicos. Tambem verifica que a funcao nao chama `.to()` para transferir tensores ao dispositivo de treino. A suite de testes foi executada no ambiente `.venv`.

A medicao representativa e a execucao end-to-end de `dp-distill` ainda precisam ser feitas com os mesmos dados e parametros do experimento; os testes unitarios nao permitem afirmar um ganho de tempo nem validar o fluxo completo.

## Referencias no codigo

- `models.py`: `TeacherEnsemble.get_token_frequency()` e `TeacherEnsemble.generate_embeddings()`.
- `utils.py`: `split_dataset_for_teachers()`.
- `main.py`: fluxo do metodo `dp-distill`.
