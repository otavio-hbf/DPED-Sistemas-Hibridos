# Analise do erro apos a distilacao e plano de otimizacao

Data: 2026-11-13
Escopo: falha no fluxo `dp-distill` ao iniciar o fine-tuning DP-SGD

## Resumo executivo

A etapa de distilacao terminou e reportou `Distillation Epoch 1 average loss: 0.3625`. A falha aconteceu depois, quando o fluxo `dp-distill` passou para o fine-tuning DP-SGD. O traceback termina no `collate` do `DPDataLoader` do Opacus, ao tentar criar um tensor vazio com `torch.zeros(..., dtype=dtype)`, e o valor recebido em `dtype` era um tipo Python, nao um `torch.dtype`.

Portanto, o erro nao aponta para a loss de distilacao, para os pesos do classificador ou para o calculo de gradientes do modelo. A causa foi reproduzida e identificada: o dataset do Hugging Face retorna cada exemplo como `dict`, mas o Opacus 1.4.0 infere os tipos de lote vazio iterando `dataset[0]`. Iterar um `dict` retorna suas chaves (`str`), e nao os tensores; no lote vazio, o Opacus tenta usar esse tipo como `dtype` de `torch.zeros`.

## Evidencias e classificacao

| Evidencia | Interpretacao |
| --- | --- |
| A distilacao imprime uma loss media antes do inicio do fine-tuning. | A falha e posterior a distilacao. |
| A excecao ocorre em `opacus/data_loader.py`, na chamada `torch.zeros(shape, dtype=dtype)`. | A excecao ocorre em `opacus/data_loader.py`, na chamada `torch.zeros(shape, dtype=dtype)`. | O erro surge ao colacionar um lote vazio do carregador privado. |
| `DPDataLoader` infere formas e dtypes iterando `dataset[0]`; o dataset original e baseado em mapeamento. | A iteracao retorna chaves de texto, cujo tipo nao pode ser usado como `torch.dtype`. |
| A reproducao local com PyTorch 2.14.1+cpu e Opacus 1.4.0 falhou com o mesmo `TypeError` para exemplos `dict`; uma amostra em tupla produziu lotes vazios com tensores `torch.int64`. | Confirma a causa no formato do dataset e o caminho de correcao. |
| O traceback original mostra Python 3.13, mas nao registra as versoes de PyTorch e Opacus. | Compatibilidade do runtime Colab ainda deve ser registrada; ela nao e necessaria para explicar a falha reproduzida localmente. |

### Mensagens que nao causaram esta excecao

- O aviso de pesos `classifier.*` nao inicializados e esperado ao criar um cabecalho de classificacao novo a partir de `bert-base-uncased`; esses pesos devem ser treinados na tarefa.
- Os avisos sobre Secure RNG, ordem do accountant, `functorch` e backward hooks sao avisos de configuracao ou manutencao. Nenhum deles corresponde ao ponto onde a execucao terminou.
- A excecao nao e um erro de privacidade reportado pelo accountant. Ainda assim, o treino nao terminou e o resultado nao deve ser tratado como um modelo DP-SGD concluido.

## Plano de diagnostico e correcao

### 1. Registrar o ambiente que falhou

O ambiente local usado para reproduzir foi Python 3.12.1, PyTorch 2.14.1+cpu, Opacus 1.4.0, Datasets 2.21.0 e Transformers 4.57.6. O ambiente Colab foi informado como Python 3.13.18, PyTorch 2.14.1+cpu e Opacus 1.4.0. Assim, PyTorch e Opacus coincidem entre os ambientes; resta validar o fluxo no runtime Python 3.13 original.

Inspecionar tambem um exemplo de `train_dataset` antes de `make_private_with_epsilon`: chaves, classe de cada valor e, para tensores, `shape` e `dtype`. Confirmar que `input_ids`, `attention_mask`, `token_type_ids` e `label`/`labels` sao tensores e que os tipos sao os esperados (IDs e mascaras inteiros; rotulos inteiros para classificacao).

### 2. Corrigir a forma de entrada do Opacus

- Adaptar os exemplos de mapeamento para tuplas antes de entregar o dataset ao `DPDataLoader`, para que o Opacus infira forma e dtype a partir dos tensores.
- Restaurar as chaves do mapeamento no loop de treino, incluindo quando o Opacus entrega um lote vazio.
- Manter o adaptador restrito ao caminho DP; os datasets e a avaliacao continuam usando o formato mapeado.
- A combinacao Python/PyTorch/Opacus do Colab foi registrada acima. Nao foi necessario alterar dependencias para corrigir o erro reproduzido.

Nao contornar o problema removendo a amostragem privada, descartando silenciosamente lotes vazios ou continuando sem DP. Essas opcoes mudariam a semantica de treino ou a garantia de privacidade.

### 3. Verificacao pequena e deterministica

O teste adicionado cobre explicitamente o caminho de lote vazio do carregador privado. Ele confirma que:

1. o lote vazio e representado por tensores com `torch.dtype` correto e formas compativeis com um exemplo real;
2. os lotes nao vazios preservam as chaves e formas esperadas pelo modelo.

O teste exercita o `DPDataLoader` real do Opacus 1.4.0, nao apenas `torch.zeros` isoladamente.

### 4. Validar o fluxo e a garantia de privacidade

O teste de integracao sintetico completou uma epoca de fine-tuning DP-SGD, avaliacao e salvamento dos checkpoints em diretorio temporario. O accountant reportou `epsilon=7.9978`, `delta=1e-05` para alvo `epsilon=8.0`, `delta=1e-05`.

Tambem foi executado o fluxo completo `dp-distill` em SST-2 com 1 professor, 4 exemplos de treino, 2 publicos, 2 de avaliacao/teste, uma epoca e lote 2. A distilacao terminou, o fine-tuning DP-SGD completou 2 passos, o accountant reportou `epsilon=7.9912`, `delta=1e-05`, e avaliacao/teste terminaram sem o erro de `dtype`. Os resultados foram gravados em um diretorio temporario, nao nos checkpoints do repositorio.

Ainda falta repetir a validacao no runtime Python 3.13.18 do Colab e, para medir desempenho/qualidade representativos, repetir com os tamanhos originais de dados.

Nao aceitar como validacao apenas a conclusao da distilacao: o fine-tuning DP-SGD tambem precisa completar. Se qualquer fallback de compatibilidade trocar o modo de gradiente do Opacus, validar novamente o modelo e o accountant no ambiente final.

## Correcao executada

Em `training.py`, foi adicionado um adaptador que apresenta amostras do dataset como tuplas somente para o carregador DP. Como o Opacus retorna lotes vazios como listas de tensores, o loop restaura as chaves do dataset antes de mover os dados para o dispositivo. O mecanismo de amostragem, clipping, ruido e accountant nao foi alterado.

Validacoes executadas no ambiente local:

- `python -m unittest discover -s tests -v`: 2 testes passaram (lote vazio e nao vazio).
- `python -m py_compile training.py tests\test_opacus_dataloader.py`: passou.
- Uma epoca DP-SGD sintetica completou treino, atualizacao do accountant, avaliacao e checkpoint.
- O fluxo completo reduzido `dp-distill` completou distilacao, fine-tuning, accountant, avaliacao e teste.
- A reproducao sem o adaptador falha com o mesmo `TypeError`; com o adaptador, o lote vazio contem tensores com `torch.int64`.

## Plano de otimizacao de desempenho

Aplicar somente depois de estabilizar e validar o carregador privado, para nao confundir mudancas de desempenho com a correcao do erro:

1. **Medir linha de base:** registrar duracao da distilacao, duracao do fine-tuning, tempo por lote, pico de memoria e versoes do ambiente. Medir a etapa que domina o tempo antes de otimizar.
2. **Ajustar o lote fisico conforme a memoria:** avaliar `max_physical_batch_size` em passos pequenos, mantendo inalterado o tamanho logico usado para a amostragem privada. Confirmar que o BatchMemoryManager nao muda o epsilon ou a semantica do treino.
3. **Reduzir custo de entrada apenas se for gargalo:** avaliar workers/pinning ou outro ajuste do DataLoader com perfil antes/depois. Revalidar os lotes vazios e a ordem/formatacao dos tensores apos cada alteracao.
4. **Preservar a metodologia:** nao alterar numero de professores, particionamento, ruido, clipping, amostragem, epsilon/delta ou acumulacao de gradientes sem uma avaliacao separada de utilidade e privacidade.
5. **Comparar resultados:** repetir com a mesma semente e dados; reportar tempo, memoria, metricas de avaliacao e epsilon/delta. Manter a otimizacao somente se houver ganho mensuravel sem regressao de corretude ou de privacidade.

## Criterios de conclusao

- O caso pequeno de `dp-sgd` e o fluxo completo de `dp-distill` terminam sem o `TypeError`.
- O caminho de lote vazio do Opacus e coberto por verificacao reproduzivel.
- O accountant reporta o epsilon e delta esperados; DP permanece habilitada durante o fine-tuning.
- As metricas e o custo de execucao sao registrados antes/depois de qualquer otimizacao.
- A combinacao de Python e dependencias usada no experimento esta documentada e fixada.

## Referencias no repositorio

- `error.txt`: traceback e avisos da execucao.
- `main.py`: distilacao seguida de fine-tuning no ramo `dp-distill`.
- `training.py`: configuracao do `PrivacyEngine`, `BatchMemoryManager` e loop DP-SGD.
- `data_loader.py`: conversao dos splits para o formato PyTorch.
- `requirements.txt`: dependencias do projeto.
- `docs/dp-sgd-opacus-fix.md`: correcao anterior de incompatibilidade de dimensao em `grad_sample`; e um problema distinto do erro de `dtype` aqui analisado.
