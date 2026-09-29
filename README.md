# Avaliação de Estratégias de Chunking para RAG (100% local)

TCC — compara **fixed-size**, **recursive** e **semantic chunking** num pipeline RAG
totalmente local (Qwen 2.5 7B gera o dataset QA, LLaMA 3.1 8B responde, Qwen 2.5 7B
julga via RAGAS, embeddings BGE-M3). Documento ativo: artigo sobre
**milho — cigarrinha-do-milho e enfezamentos** (`Textos_exemplo/milho_enfezamento.pdf`).

## Estrutura

```text
├── src/
│   ├── config.py               # TUDO que varia: documento, modelos, chunking, top-k
│   ├── common.py               # PDF genérico, loaders 4 bits, embeddings, artefatos
│   ├── 01_gerar_dataset_qa.py  # Fase 0 → Textos_exemplo/qa_dataset_<doc>.json
│   ├── 02_gerar_respostas.py   # Fase 1 (LLaMA) → analises/geracoes_<doc>_<estrategia>.json
│   └── 03_avaliar_ragas.py     # Fase 2 (Qwen) → analises/ragas_{results,summary}_<doc>.csv
├── implementacoes/             # chunkers customizados (fixed, recursive, semantic)
├── Textos_exemplo/             # PDFs + datasets QA gerados
├── analises/                   # artefatos gerados (NÃO versionados, ver .gitignore)
├── notebooks/tcc.ipynb         # notebook enxuto: só setup + chamadas + gráficos
└── scripts/legado/             # scripts antigos (referência; o fluxo oficial é src/)
```

## Fluxo (1 processo por fase — cada fase salva em disco e o processo morre)

```bash
# Fase 0 — dataset QA (GPU, Qwen 2.5 7B)
python src/01_gerar_dataset_qa.py --doc milho

# Fase 1 — geração RAG (GPU, LLaMA 3.1 8B). Teste pequeno antes do total:
python src/02_gerar_respostas.py --doc milho --amostra 5
python src/02_gerar_respostas.py --doc milho --amostra todas

# Fase 2 — RAGAS (GPU, Qwen 2.5 7B). Rode em sessão/processo NOVO (VRAM limpa):
python src/03_avaliar_ragas.py --doc milho
```

Trocar de documento: `--doc soja` (ou mude `DOC_ATIVO` em `src/config.py`).
Todos os caminhos de saída derivam do `--doc`, então experimentos não se sobrescrevem.

## No Colab (só clone + células)

1. Abra `notebooks/tcc.ipynb` no Colab (via GitHub).
2. Siga as células: GPU check → env VRAM → install → login HF (via `getpass`,
   nunca hardcoded) → Fase 0 → Fase 1 → **reinicie a sessão** → Fase 2 → gráficos.
3. Nunca commite token: use Colab Secrets ou `getpass`. Se um token vazar,
   revogue em huggingface.co → Settings → Tokens.

## Notas de VRAM (T4 15 GB)

- Embeddings BGE-M3 rodam na **CPU** (`EMBEDDINGS_DEVICE` no config) — libera ~2,3 GB.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` antes de qualquer uso de GPU.
- Cada fase imprime `[VRAM ...]` para diagnóstico.
- Se a Fase 2 cair, os JSONs da Fase 1 continuam em `analises/` — reexecute só a Fase 2.
