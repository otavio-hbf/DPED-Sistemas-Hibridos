# Notebook de execução do ataque ALGEN sobre DPED

Arquivo: [attack_algen_dped.ipynb](../../notebooks/attack_algen_dped.ipynb)

Alternativa para a metodologia de ruido calibrado: [attack_algen_dped_calibracao_ruido.ipynb](../../notebooks/attack_algen_dped_calibracao_ruido.ipynb). Ambas as variantes clonam explicitamente a branch `robust-metodol`, que contém o helper `preflight_conditions` e o código de proveniência necessário. A variante de calibração executa `--method dp-distill --embedding_privacy_sweep --delta 1e-5`, sem `--epsilon` ou DP-SGD downstream. Antes de executar, coloque no Drive os pesos `.pt` do decodificador ALGEN congelado. A configuração atual mapeia explicitamente o caminho esperado `checkpoint_epoch_99.pt` para o arquivo informado `checkpoint_epoch_100.pt`; confirme que esse arquivo é o checkpoint que deseja usar.

Antes da grade SST-2, a variante de calibração inclui um sanity test end-to-end reduzido em CoLA (64 exemplos, 2 teachers, 1 época), seguido por verificações de quatro exports: vetores finitos, alinhamento de IDs/labels e epsilon correto nos metadados. As saídas ficam em `glue_cola_*`, separadas da grade SST-2, e não devem ser usadas como resultados do ataque. Para usar os novos controles, a branch carregada no Colab precisa conter estas alterações. Ambas as variantes agora usam o mesmo preflight de proveniência.

Sequência de execução no Colab, no padrão de [dped_medium.ipynb](../../notebooks/dped_medium.ipynb):

1. **Montagem do Drive e clone do repositório.** `git pull` se o repositório já existir, senão `git clone`. Depois `pip install -r requirements.txt evaluate rouge-score pot wandb`. Os últimos quatro pacotes são necessários para importar o ALGEN (`evaluate` em `decoder_finetune_trainer.py`, `rouge-score` em `eval_metrics.py`, `pot` nos métodos de alinhamento e `wandb` em `decoder_finetune_trainer.py`), mas não estão no `requirements.txt` da raiz.
2. **Export pré-downstream com `--embedding_privacy_sweep`.** A grade gera os exports para ε∈{1,2,4,8}, sem DP-SGD downstream e com o vocabulário completo fixo. O atacante usa os IDs exportados exceto tokens especiais; não depende de `token_freq.npy`. As quatro tabelas, se divulgadas juntas, requerem composição de privacidade.

3. **Preflight e smoke test.** O preflight descarta módulos de proveniência antigos do cache Python, importa novamente o atacante do caminho configurado e confere hashes, metadados e proveniência das cinco condições. Se a função estiver ausente no arquivo em disco, exibe caminho e revisão Git. Os notebooks fazem checkout explícito da branch `robust-metodol`; não use a branch `main` nem `calibracao-de-ruido`, que não contêm esse helper. O smoke usa 10 amostras de treino e 10 de teste e é identificado por `--run_kind smoke`. Exports antigos sem a proveniência exigida não passam no preflight. Consulte [auditoria-proveniencia-fase0.md](auditoria-proveniencia-fase0.md).
4. **Grade de ataque.**
   - DPED: ε ∈ {1, 2, 4, 8} × `train_samples` ∈ {1, 10, 30, 100, 300, 500, 1000} × ridge. Saída em `results/attack/dped_eps<tag>/n<N>_r<R>/`.
   - Baseline `bert_pretrained`: mesma grade de `train_samples` e ridge. Usa o export de ε=8.0 para os IDs de tokens. Saída em `results/attack/bert_pretrained/`. As execuções da grade são identificadas por `--run_kind research`.
5. **Persistência incremental no Drive.** Cada execução recebe um diretório próprio `SIH/privEmbeddings/<RUN_ID>/`. Após o sweep de export, as tabelas (incluindo hashes e snapshots) são copiadas para `exports/research/`; o preflight e o smoke são salvos logo ao terminar; cada execução da grade é copiada para `results_attack/` antes de iniciar a próxima. Na variante de calibração, os exports reduzidos de CoLA também são copiados para `exports/sanity/` e permanecem separados da pesquisa SST-2. A cópia final repete a sincronização como verificação. Erros de geração interrompem a célula sem copiar artefatos incompletos.

## Decisões e limites

- `ALGEN_CKPT` precisa ser ajustado para a pasta do checkpoint do decodificador ALGEN no Drive. O checkpoint é congelado (sem retreino, ver P5).
- O ridge está fixo em `1.0`. A escolha por validação ou fixa é a decisão P6, ainda em aberto. Ajustar `RIDGE_VALUES` depois dessa decisão.
- A grade roda por `subprocess` com os mesmos argumentos de linha de comando, em vez de uma célula `!python` por execução. Assim a grade completa cabe em uma célula.
- O CSV do atacante usa a coluna `source`, e não `method` como descrito no plano. Não há colunas BLEU no CSV atual, só ROUGE e cosseno. Esses dois pontos precisam ser alinhados antes da análise final.
- Nada disso foi executado localmente (sem `torch`, `transformers` e `datasets` no ambiente). A validação é o smoke test no Colab.
