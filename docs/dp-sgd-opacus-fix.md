# DPED: Investigacao e correcao do erro DP-SGD (Opacus)

Data: 2026-11-05
Escopo: pipeline DPED com fine-tuning DP-SGD

## Contexto

Durante o treinamento com DP-SGD, a execucao falhava no primeiro `optimizer.step()` com erro de empilhamento interno no Opacus:

- RuntimeError: stack expects each tensor to be equal size, but got [8] at entry 0 and [1] at entry 1

## Causa raiz

O erro foi causado por desalinhamento na dimensao de amostras dos `grad_sample` entre parametros diferentes no clipping por amostra do Opacus.

No caminho BERT + Opacus em `grad_sample_mode="functorch"`, quando `position_ids` nao e passado explicitamente no forward, a camada de posicao pode operar com shape broadcastado (ex.: `[1, seq_len]`).

Isso pode produzir `grad_sample` da embedding posicional com batch efetivo 1, enquanto outros parametros ficam com batch real (ex.: 8). Ao calcular normas por parametro e empilhar (`torch.stack`), ocorre incompatibilidade `[8]` vs `[1]`.

## Componentes afetados

- Forward de treino e avaliacao em training.py
- Wrapping do modelo no Opacus em training.py
- Fluxo de execucao do metodo dp-distill em main.py (caminho que cai no fine-tuning DP-SGD)

## Correcao aplicada (minima)

Arquivo alterado: training.py

1. Passagem explicita de `position_ids` batched no forward quando o modelo suporta esse argumento.
2. Garantia de `model.train()` antes de `PrivacyEngine.make_private_with_epsilon(...)` para satisfazer validacao estrita do Opacus.
3. Instrumentacao diagnostica temporaria para imprimir metadados de `grad_sample` somente se o mesmo erro voltar.

A correcao preserva o algoritmo DP-SGD original e nao altera arquitetura, parametros treinaveis, clipping, ruido gaussiano ou budget de privacidade.

## Evidencias de validacao

Ambiente usado:

- Python da venv do projeto: `.venv/Scripts/python.exe`
- torch: 2.14.1+cpu
- opacus: 1.4.0

Teste 1 (caminho DP-SGD direto):

- Comando: `main.py --method dp-sgd --dataset glue --task sst2 --num_epochs 1 --batch_size 16 --max_train_samples 32 --max_eval_samples 32 --max_test_samples 32 --max_public_samples 32`
- Resultado:
  - completou pelo menos um `optimizer.step()`
  - completou 1 epoca
  - clipping executou sem erro
  - accountant atualizou epsilon (`Privacy spent after epoch 1: e=7.9912, d=1e-05`)
  - avaliacao final executou

Teste 2 (caminho original que falhava: DPED completo):

- Comando: `main.py --method dp-distill --dataset glue --task sst2 --num_epochs 1 --batch_size 16 --max_train_samples 32 --max_eval_samples 32 --max_test_samples 32 --max_public_samples 32`
- Resultado:
  - distillation completou
  - fine-tuning DP-SGD completou
  - sem erro de stack `[8]` vs `[1]`
  - epsilon atualizado apos treino
  - avaliacao e teste finais executados

## Avisos observados e classificacao

Os avisos abaixo foram analisados e nao sao a causa direta do crash corrigido:

- Secure RNG turned off
- Optimal RDP order is the largest alpha
- Warnings de deprecacao do functorch
- Warnings de non-full backward hook

Esses avisos podem demandar manutencao futura, mas nao invalidam a correcao do erro de batch-dimension nos `grad_sample`.

## Implicacoes de privacidade

A correcao NAO invalida o mecanismo de DP-SGD:

- gradientes por amostra permanecem ativos
- clipping por amostra permanece ativo
- ruido gaussiano permanece ativo
- budget alvo permanece `epsilon=8.0` e `delta=1e-5`

## Referencias no codigo

- training.py
- main.py
- models.py

## Tutorial rapido de command lines

Esta secao mostra como executar os principais cenarios via linha de comando.

### 1) Preparar ambiente

No PowerShell, execute na raiz do projeto:

```powershell
.\.venv\Scripts\Activate.ps1
python -V
pip install -r requirements.txt
```

### 2) Estrutura base do comando

Formato geral:

```powershell
python main.py --dataset glue --task sst2 --method <metodo> [opcoes]
```

Metodos suportados hoje:

- `non-private`
- `dp-sgd`
- `pate-distill`
- `dp-distill`

### 3) Cenarios principais

Treino baseline sem privacidade:

```powershell
python main.py --dataset glue --task sst2 --method non-private --num_epochs 1 --batch_size 8
```

Treino com DP-SGD:

```powershell
python main.py --dataset glue --task sst2 --method dp-sgd --epsilon 8.0 --delta 1e-5 --num_epochs 1 --batch_size 8
```

DP-Distill completo (destilacao + fine-tune com DP-SGD):

```powershell
python main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0 --num_epochs 1 --batch_size 8
```

DP-Distill sem DP no fine-tune (destilacao ativa + fine-tune sem DP-SGD):

```powershell
python main.py --dataset glue --task sst2 --method dp-distill --no_dp --num_epochs 1 --batch_size 8
```

PATE-Distill sem DP no fine-tune:

```powershell
python main.py --dataset glue --task sst2 --method pate-distill --no_dp --num_epochs 1 --batch_size 8
```

### 4) Rodar com subconjunto pequeno (debug rapido)

Para validar pipeline sem custo alto:

```powershell
python main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0 --num_epochs 1 --batch_size 8 --max_train_samples 32 --max_eval_samples 32 --max_test_samples 32 --max_public_samples 32 --subset_strategy random --subset_seed 42
```

### 5) Onde ver resultados

- Checkpoints: pasta `models/`
- Resultados consolidados: pasta `results/`
- Metricas finais: impressas no terminal ao final da execucao

## Lista de exemplos uteis

### GLUE (SST-2)

```powershell
# Baseline sem DP
python main.py --dataset glue --task sst2 --method non-private --num_epochs 3 --batch_size 16

# DP-SGD com epsilon menor (mais privacidade)
python main.py --dataset glue --task sst2 --method dp-sgd --epsilon 4.0 --num_epochs 3 --batch_size 16

# DP-Distill com mult-layer noise desativado (ablation)
python main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0 --no_multi_layer_noise --num_epochs 3
```

### Outras tasks GLUE

```powershell
python main.py --dataset glue --task qqp --method dp-distill --epsilon 8.0
python main.py --dataset glue --task mnli --method dp-distill --epsilon 8.0
python main.py --dataset glue --task cola --method dp-distill --epsilon 8.0
```

### CoNLL-2003

```powershell
python main.py --dataset conll2003 --method dp-sgd --epsilon 8.0
python main.py --dataset conll2003 --method dp-distill --epsilon 8.0
```

### Comparacao automatica de metodos

```powershell
python main.py --dataset glue --task sst2 --run_all_methods --epsilon 8.0
```

### Curva privacidade-utilidade

```powershell
python main.py --dataset glue --task sst2 --privacy_utility_tradeoff
```

### Reprodutibilidade

```powershell
python main.py --dataset glue --task sst2 --method dp-distill --epsilon 8.0 --seed 42
```
