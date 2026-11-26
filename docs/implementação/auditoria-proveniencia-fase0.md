# Fase 0: proveniência e reprodutibilidade

## Situação

Os controles para novas execuções foram implementados, mas a Fase 0 **não está
validada**: os exports e os arquivos brutos históricos estão no ambiente do
experimento e não foram examinados aqui. A execução no Colab e a recuperação
do PDF-fonte, dos resumos e dos CSVs históricos são necessárias antes de
considerar qualquer número histórico rastreável.

## Geração dos exports

Cada novo export DPED pré-downstream registra no `metadata.txt` o SHA-256 dos
arquivos NumPy, os parâmetros efetivos do mecanismo e, quando disponíveis, as
revisões do modelo e tokenizer. `generation_config.json` registra os valores
efetivos de `Config` para o epsilon exportado. No início do treino são
capturados revisão Git, hashes dos arquivos Python relevantes, alterações no
código e versões de pacotes; no export são gravados
`generation_provenance.json`, `generation_code.zip` e
`generation_requirements.observed.txt`. O ZIP contém o código local cujo hash
foi capturado; se ele mudar durante a execução, o export falha.

O arquivo de requisitos lista as versões instaladas observadas, **não** é um
lock resolvido e verificado por hashes de pacotes. É preciso gerar e validar um
lock apropriado ao runtime efetivo do Colab antes de declarar o ambiente
integralmente reproduzível. Quando uma revisão de modelo ou tokenizer não é
disponibilizada pela biblioteca, ela aparece como `None`; não foi inferida.
O ruído usa entropia do sistema operacional: semente e configuração não
reproduzem a realização do ruído. Preserve a matriz efetivamente exportada.

## Ataque e grade

Cada execução registra em `summary.json` os hashes da tabela, dos IDs, dos
metadados e dos arquivos de geração, a configuração de geração disponível, o
checkpoint selecionado e o hash da configuração do decoder. O checkpoint e sua
configuração são conferidos antes e depois do ataque. Registra também hashes e
IDs das frases selecionadas e fingerprints dos splits carregados.
`attack_provenance.json`, `attack_code.zip` e
`attack_requirements.observed.txt` registram argumentos, revisão Git, código
local e versões observadas no início do ataque.

O preflight de ambos os notebooks verifica as quatro condições e o BERT:
matrizes distintas, IDs iguais, epsilon e parâmetros presentes no `metadata.txt`,
hashes dos arquivos e existência dos artefatos de geração. O ataque de pesquisa
repete a verificação de metadados e exige os hashes que o preflight observou,
inclusive o hash combinado dos metadados e dos arquivos de geração. Exports
antigos sem esses arquivos falham no preflight: não invente sua proveniência;
recupere os dados originais ou gere novos exports. Smoke tests continuam
identificados e separados dos resultados de pesquisa.

O procedimento de teste adotado mantém o comportamento do código: primeiras
frases elegíveis de `validation`, na ordem do dataset. O alinhamento usa
amostragem sem reposição do `train`, com a semente informada. As grades
comparam os hashes dos pares de treino para cada `(n, seed)` e os hashes das
frases de teste entre condições; divergências interrompem a execução. O
notebook de análise rejeita hashes de seleção divergentes quando disponíveis e
avisa quando não é possível verificá-los.

O relatório histórico descreve 200 frases amostradas, mas os arquivos brutos e
o código efetivamente executado ainda não foram recuperados. Não se pode
estabelecer qual procedimento produziu aquele relatório apenas com base no
código atual.

## Divergência histórica sobre tokens

O documento antigo [export-pre-downstream-vs-save-model.md](../analises/export-pre-downstream-vs-save-model.md)
descreve uma seleção restrita a tokens do conjunto público. A documentação
[otimizacao-contagem-tokens.md](otimizacao-contagem-tokens.md) descreve a
contagem antiga de presença dos tokens nos conjuntos dos professores.
Ambos registram um fluxo histórico. Hoje, a liberação e exportação usam o
vocabulário fixo completo, sem contagens privadas; o ataque exclui apenas
tokens especiais. Exports antigos não devem ser presumidos equivalentes aos
atuais sem inspeção dos seus arquivos e código de origem.

## Inventário histórico no Colab

Após montar o Drive e disponibilizar esta versão do repositório, ajuste os
caminhos e execute:

```python
!python algen/SIH-ALGEN-main/src/audit_historical_artifacts.py \
  --root "/content/drive/MyDrive/SIH/privEmbeddings/2026-11-16" \
  --output "/content/inventario_historico.json"
```

O inventário registra caminhos, tamanhos e SHA-256 de PDFs, resumos, CSVs por
sentença, matrizes, IDs e metadados encontrados. Não copia nem altera os
originais e não sobrescreve o manifesto. Ele não identifica automaticamente
qual PDF ou export foi efetivamente usado; confira a origem, a cobertura e os
pares resumo/CSV manualmente. Sem os arquivos históricos não é possível
recalcular a célula contaminada ou validar as conclusões anteriores.
