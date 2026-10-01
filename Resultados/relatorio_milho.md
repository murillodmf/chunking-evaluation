# Milho — consolidado RAGAS (A100)

Fonte: `Resultados/saída1.txt` (log Fase 2) + `Resultados/tabela_consolidada_milho.csv`.
Gerador: `meta-llama/Llama-3.1-8B-Instruct` 4-bit. Juiz: `Qwen/Qwen2.5-14B-Instruct` 4-bit,
greedy, `max_new_tokens=1024`, chat-template via `src/common.py:build_hf_llm`.
Embeddings: `BAAI/bge-m3` em CPU. QA: 15 pares curados (`Textos_exemplo/qa_milho_curado.json`).
`TOP_K=3`, `CHUNK_SIZE=512`, `OVERLAP=51`.

## Tabela (médias; n_valid entre parênteses)

| estratégia | chunks | faithfulness | answer_relevancy | answer_correctness | context_precision | context_recall |
|---|---|---|---|---|---|---|
| fixed | 46 | 0.655 (14/15) | 0.445 (15/15) | 0.464 (13/15) | 0.744 | 0.583 |
| recursive | 15 | **0.819** | 0.556 (15/15) | 0.548 (14/15) | **0.933** | **0.822** |
| semantic (p50) | 70 | 0.774 (14/15) | **0.607** (15/15) | **0.575** (14/15) | 0.856 | 0.617 |
| baseline (sem retrieval) | — | — | 0.240 (15/15) | 0.138 (14/15) | — | — |
