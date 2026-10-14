# Pacote VIM3: como rodar

Três binários aarch64 estáticos, todos compilados a partir deste diretório:

| Binário | O que é |
|---|---|
| `bench_int16_compact_aarch64` | QNN Q7.8 com armazenamento int16, a mesma aritmética do C certificado |
| `bench_int64_certified_aarch64` | O C certificado (armazenamento int64), a fonte que o ESBMC verificou |
| `bench_float32_baseline_aarch64` | A rede float32 sem quantizar. É a linha de base e não é verificada |

Os dados são só do split de validação: 7.830 recortes em `images_all.bin` e 1 recorte em `images_one.bin`. Os logits esperados vêm do C certificado, executado no host.

## Rodar

A partir do host:

```sh
adb push output/vim3_bundle_20260928 /data/local/tmp/vim3
adb shell 'cd /data/local/tmp/vim3 && sh run_on_vim3.sh'
adb pull /data/local/tmp/vim3/results_vim3.jsonl /data/local/tmp/vim3/board_vim3.txt .
```

Se a placa estiver rodando Linux em vez de AAOS, copie o diretório e rode `sh run_on_vim3.sh` dentro dele.

O script roda três vezes para cada variante:
1. Consistência: 1 passada sobre todas as imagens.
2. Latência: 3 passadas.
3. Memória: 1 imagem, 1000 passadas.

Cada rodada grava uma linha JSON em `results_vim3.jsonl`. Os dados da placa vão para `board_vim3.txt`: kernel, CPU, fingerprint, governor e sha256 dos binários.

## Como ler

- **`mismatches`** deve ser 0.
  - Nas variantes QNN, ele compara os 43 logits int64 exatos com a referência do host.
  - Na float32, compara só a classe prevista com a execução float do host.
  - O binário sai com código 1 quando há divergência.
- **`accuracy`**: no host, deu 0,902427 nas QNN e 0,903576 na float32.
- **`latency_us_*`** mede encode + forward por imagem com CLOCK_MONOTONIC. Não inclui a leitura dos arquivos.
- **`peak_rss_kib`** é o VmHWM do próprio processo. Use a rodada `*_rss_one_image`: nas outras, o arquivo de 7.830 imagens (~72 MB) domina o RSS.
- **`accuracy` 0 na rodada `*_rss_one_image`** é esperado: o 1º recorte de validação é mal classificado pelas três redes. Essa rodada serve só para RSS e latência.
- **Tamanho do binário**: veja `bundle_manifest.json` (saída do `size`).

Para a latência ser comparável, anote o governor da CPU e não rode nada em paralelo. Se possível, deixe o governor em `performance` (precisa de root).
