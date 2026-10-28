"""Fase 1 — geração RAG com LLaMA 3.1 8B + braço baseline sem retrieval.

Roda em processo próprio: ao terminar, salva geracoes_<doc>_<estrategia>.json
em analises/ e o processo morre (VRAM 100% limpa para a Fase 2).

Braços: fixed / recursive / semantic (RAG) + baseline (LLM puro, sem contexto).
O baseline permite a Tabela 2 do TCC: RAG (melhor chunking) x LLM paramétrico.

Uso:
    python src/02_gerar_respostas.py --doc milho --amostra 5
    python src/02_gerar_respostas.py --doc milho --amostra todas
    python src/02_gerar_respostas.py --doc milho --amostra todas --sem-baseline
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))
import config
import common

RAG_SYSTEM = """Você é um assistente virtual agronômico de alta precisão.
Responda à pergunta do usuário baseando-se estritamente nas informações fornecidas no contexto abaixo.
Se a resposta não estiver contida no contexto, diga "Não encontrei essa informação no contexto".
Não invente nenhum fato ou valor numérico que não esteja explicitamente escrito no contexto."""

BASELINE_SYSTEM = """Você é um assistente virtual agronômico de alta precisão.
Responda à pergunta do usuário com base no seu próprio conhecimento, de forma
direta e factual. Se não souber a resposta, diga "Não sei responder com segurança"."""


def rag_prompt(context_text: str, question: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{RAG_SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>
Contexto:
{context_text}

Pergunta:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def baseline_prompt(question: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{BASELINE_SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>
Pergunta:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Fase 1: geração RAG (LLaMA 3.1 8B).")
    ap.add_argument("--doc", default=config.DOC_ATIVO, choices=list(config.DOCS))
    ap.add_argument("--corpus", action="store_true",
                    help="Usa o benchmark multi-documento (10 PDFs de Documentos/Dataset)")
    ap.add_argument("--docs", default="",
                    help="Com --corpus: lista de doc-ids separada por vírgula (default: todos)")
    ap.add_argument("--amostra", default="todas",
                    help="N de perguntas ou 'todas' (default: %(default)s)")
    ap.add_argument("--k", type=int, default=config.TOP_K, help="Top-K recuperados")
    ap.add_argument("--k-list", default="",
                    help="Lista de valores de K separados por vírgula para rodar numa única sessão (ex: '5,7' ou '3,5,7')")
    ap.add_argument("--percentile-list", default="",
                    help="Lista de percentis do Semantic separados por vírgula (ex: '40,50,70')")
    ap.add_argument("--estrategias", default="fixed,recursive,semantic",
                    help="Chunkers a indexar/rodar (default: %(default)s)")
    ap.add_argument("--index-only", action="store_true",
                    help="Só extrai e chunka, sem embeddings/LLM (valida o corpus)")
    ap.add_argument("--sem-baseline", action="store_true",
                    help="Pula o braço baseline (LLM puro, sem retrieval)")
    args = ap.parse_args()

    if args.k_list.strip():
        k_values = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    else:
        k_values = [args.k]

    percentiles_custom = [
        float(x.strip()) for x in args.percentile_list.split(",") if x.strip()
    ] if args.percentile_list.strip() else []

    common.setup_reproducibility()
    if args.corpus:
        base_paths = config.corpus_paths()
        doc_label = "corpus"
    else:
        base_paths = config.paths(args.doc)
        doc_label = args.doc

    print("=" * 60)
    print(f" FASE 1: GERAÇÃO RAG — doc='{doc_label}' | K={k_values}")
    print("=" * 60)

    qa = common.select_sample(common.load_json(base_paths["qa"]), args.amostra)
    print(f"Dataset QA: {len(qa)} perguntas.")

    t0_pdf = time.perf_counter()
    if args.corpus:
        doc_filter = [d.strip() for d in args.docs.split(",") if d.strip()] or None
        texts = common.load_corpus_texts(doc_filter)
        total_chars = sum(len(t) for t in texts.values())
        print(f"Corpus: {len(texts)} documentos, {total_chars} caracteres.")
    else:
        doc_cfg = config.DOCS[args.doc]
        text = common.load_and_clean_pdf(str(base_paths["pdf"]), doc_cfg)
        total_chars = len(text)
        print(f"Texto: {total_chars} caracteres.")
    tempo_extracao_pdf_s = round(time.perf_counter() - t0_pdf, 4)
    print(f"[tempo] Extração e limpeza de PDF(s): {tempo_extracao_pdf_s:.3f}s")

    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from fixed_size_chunker import MeuFixedSizeChunkerTCC
    from recursive_chunker import MeuRecursiveChunkerTCC
    from semantic_chunker import MeuSemanticChunkerTCC

    raw_strategy_names = [s.strip() for s in args.estrategias.split(",") if s.strip()]
    strategy_names = []
    for s in raw_strategy_names:
        if s == "semantic" and percentiles_custom:
            for p in percentiles_custom:
                p_lbl = int(p) if float(p).is_integer() else p
                s_name = "semantic" if float(p) == float(config.SEMANTIC_PERCENTILE) else f"semantic_p{p_lbl}"
                if s_name not in strategy_names:
                    strategy_names.append(s_name)
        else:
            if s not in strategy_names:
                strategy_names.append(s)

    def _parse_sem_percentile(strat_name: str) -> float | None:
        if strat_name == "semantic":
            return float(config.SEMANTIC_PERCENTILE)
        if strat_name.startswith("semantic_p"):
            try:
                return float(strat_name[len("semantic_p"):])
            except ValueError:
                return None
        return None

    def _parse_token_chunker(strat_name: str) -> tuple[str, int, int] | None:
        if strat_name in ("fixed", "recursive"):
            return strat_name, config.CHUNK_SIZE, config.CHUNK_OVERLAP
        for base in ("fixed_", "recursive_"):
            if strat_name.startswith(base):
                try:
                    cs = int(strat_name[len(base):])
                    return base[:-1], cs, max(1, cs // 10)
                except ValueError:
                    return None
        return None

    unknown = [
        s for s in strategy_names
        if _parse_token_chunker(s) is None and _parse_sem_percentile(s) is None
    ]
    if unknown:
        raise SystemExit(
            f"--estrategias desconhecidas: {unknown}. Use fixed, recursive, fixed_<N>, recursive_<N>, semantic ou semantic_p<N>."
        )

    def _make_chunkers(emb):
        out = {}
        for s in strategy_names:
            tc = _parse_token_chunker(s)
            if tc is not None:
                kind, cs, co = tc
                if kind == "fixed":
                    out[s] = MeuFixedSizeChunkerTCC(chunk_size=cs, chunk_overlap=co)
                else:
                    out[s] = MeuRecursiveChunkerTCC(chunk_size=cs, chunk_overlap=co)
            else:
                pct = _parse_sem_percentile(s)
                out[s] = MeuSemanticChunkerTCC(
                    embeddings=emb, percentile_threshold=pct)
        return out

    if args.index_only:
        needs_emb = any(_parse_sem_percentile(s) is not None for s in strategy_names)
        emb_idx = common.build_embeddings() if needs_emb else None
        chunkers = _make_chunkers(emb_idx)

        for name in strategy_names:
            splitter = chunkers[name]
            if args.corpus:
                counts = {d: len(splitter.split_text(t)) for d, t in texts.items()}
                detail = ", ".join(f"{d}={n}" for d, n in counts.items())
                print(f"-> {name.upper()}: {sum(counts.values())} chunks ({detail}).")
            else:
                print(f"-> {name.upper()}: {len(splitter.split_text(text))} chunks.")
        print("--index-only: extração e chunking validados; saindo sem embeddings/LLM.")
        return

    embeddings = common.build_embeddings()  # CPU por default (config)
    chunkers = _make_chunkers(embeddings)

    from transformers import pipeline as _  # noqa: F401 (garante backend carregado)
    dbs = {}
    chunk_stats = {}
    for name in strategy_names:
        splitter = chunkers[name]
        t0_chunk = time.perf_counter()
        t_standalone = 0.0
        if args.corpus:
            docs, counts = [], {}
            for doc_id, doc_text in texts.items():
                chunks = splitter.split_text(doc_text)
                t_standalone += getattr(splitter, "last_standalone_seconds", 0.0)
                counts[doc_id] = len(chunks)
                docs.extend(
                    Document(page_content=c, metadata={"strategy": name, "doc_id": doc_id})
                    for c in chunks
                )
            t_wall = time.perf_counter() - t0_chunk
            t_chunk = t_standalone if t_standalone > 0 else t_wall
            detail = ", ".join(f"{d}={n}" for d, n in counts.items())
            print(f"-> {name.upper()}: {len(docs)} chunks em {t_wall:.3f}s (custo standalone={t_chunk:.3f}s) ({detail}).")
            collection = f"db_corpus_{name}"
        else:
            chunks = splitter.split_text(text)
            t_wall = time.perf_counter() - t0_chunk
            t_standalone = getattr(splitter, "last_standalone_seconds", 0.0)
            t_chunk = t_standalone if t_standalone > 0 else t_wall
            print(f"-> {name.upper()}: {len(chunks)} chunks em {t_wall:.3f}s (custo standalone={t_chunk:.3f}s).")
            docs = [
                Document(page_content=c, metadata={"strategy": name, "doc_id": args.doc})
                for c in chunks
            ]
            collection = f"db_{args.doc}_{name}"
        t0_idx = time.perf_counter()
        dbs[name] = Chroma.from_documents(docs, embeddings, collection_name=collection)
        t_idx = time.perf_counter() - t0_idx
        print(f"   [indexação Chroma {name.upper()}] {t_idx:.3f}s ({len(docs)} chunks)")
        chunk_stats[name] = {
            "n_chunks": len(docs),
            "tempo_chunking_s": round(t_chunk, 4),
            "tempo_indexacao_s": round(t_idx, 4),
        }
    print("Indexação concluída.")

    pipe, tokenizer, gen_kwargs = common.build_causal_llm(
        config.MODELS["rag_gerador"], config.RAG_TOKENS, config.RAG_TEMP)
    llm = common.build_hf_llm(pipe, tokenizer, gen_kwargs, use_chat_template=False)

    timing_rows = []
    for k_val in k_values:
        k_paths = config.corpus_paths(k=k_val) if args.corpus else config.paths(args.doc, k=k_val)
        print("\n" + "-" * 60)
        print(f" RODANDO CONSULTAS E GERAÇÃO PARA TOP_K = {k_val}")
        print("-" * 60)
        for strategy, db in dbs.items():
            print(f"\nGerando respostas: {strategy.upper()} (K={k_val})")
            records = []
            per_ctx = max(500, config.RAG_MAX_CONTEXT_CHARS // max(1, k_val))
            t_search_total = 0.0
            t_gen_total = 0.0
            for i, item in enumerate(qa):
                t0_s = time.perf_counter()
                ctx_docs = db.similarity_search(item["question"], k=k_val)
                t_search_total += time.perf_counter() - t0_s
                # Trava anti-OOM: cada contexto e truncado; o prompt nunca explode
                # mesmo que um chunk gigante seja recuperado.
                contexts = [d.page_content[:per_ctx] for d in ctx_docs]
                retrieved_docs = [d.metadata.get("doc_id") for d in ctx_docs]
                context_text = "\n\n".join(contexts)
                t0_g = time.perf_counter()
                try:
                    ans = llm.invoke(rag_prompt(context_text, item["question"]))
                except Exception as e:
                    print(f"  Erro na pergunta {i + 1}: {e}")
                    ans = "Erro na geração da resposta."
                t_gen_total += time.perf_counter() - t0_g
                records.append({
                    "question": item["question"], "contexts": contexts,
                    "retrieved_docs": retrieved_docs,
                    "answer": ans, "ground_truth": item["ground_truth"],
                })
                print(f"  Pergunta {i + 1}/{len(qa)} processada.")
            common.save_json(records, k_paths["geracoes"](strategy))
            n_q = max(1, len(qa))
            row_t = {
                "doc": doc_label,
                "strategy": strategy,
                "top_k": k_val,
                "semantic_percentile": _parse_sem_percentile(strategy),
                "chunk_size": _parse_token_chunker(strategy)[1] if _parse_token_chunker(strategy) else None,
                "n_chunks": chunk_stats[strategy]["n_chunks"],
                "tempo_chunking_s": chunk_stats[strategy]["tempo_chunking_s"],
                "tempo_indexacao_s": chunk_stats[strategy]["tempo_indexacao_s"],
                "tempo_busca_total_s": round(t_search_total, 4),
                "tempo_busca_medio_ms": round((t_search_total / n_q) * 1000.0, 2),
                "tempo_geracao_llm_total_s": round(t_gen_total, 4),
                "tempo_geracao_llm_medio_s": round(t_gen_total / n_q, 3),
                "tempo_consulta_total_medio_s": round((t_search_total + t_gen_total) / n_q, 3),
            }
            timing_rows.append(row_t)
            print(
                f"  [tempo {strategy.upper()} K={k_val}] "
                f"chunk={row_t['tempo_chunking_s']:.2f}s | "
                f"index={row_t['tempo_indexacao_s']:.2f}s | "
                f"busca_média={row_t['tempo_busca_medio_ms']:.1f}ms | "
                f"geração_média={row_t['tempo_geracao_llm_medio_s']:.2f}s/pergunta"
            )

    # Braço baseline: mesmo LLM, mesmas perguntas, ZERO contexto recuperado.
    # Mede o conhecimento paramétrico puro (Tabela 2 do TCC: RAG x LLM).
    if not args.sem_baseline:
        print("\nGerando respostas: BASELINE (sem retrieval)")
        records = []
        t_gen_base = 0.0
        for i, item in enumerate(qa):
            t0_g = time.perf_counter()
            try:
                ans = llm.invoke(baseline_prompt(item["question"]))
            except Exception as e:
                print(f"  Erro na pergunta {i + 1}: {e}")
                ans = "Erro na geração da resposta."
            t_gen_base += time.perf_counter() - t0_g
            records.append({
                "question": item["question"], "contexts": [],
                "retrieved_docs": [],
                "answer": ans, "ground_truth": item["ground_truth"],
            })
            print(f"  Pergunta {i + 1}/{len(qa)} processada.")
        common.save_json(records, base_paths["geracoes"]("baseline"))
        n_q = max(1, len(qa))
        timing_rows.append({
            "doc": doc_label,
            "strategy": "baseline",
            "top_k": 0,
            "semantic_percentile": None,
            "chunk_size": None,
            "n_chunks": 0,
            "tempo_chunking_s": 0.0,
            "tempo_indexacao_s": 0.0,
            "tempo_busca_total_s": 0.0,
            "tempo_busca_medio_ms": 0.0,
            "tempo_geracao_llm_total_s": round(t_gen_base, 4),
            "tempo_geracao_llm_medio_s": round(t_gen_base / n_q, 3),
            "tempo_consulta_total_medio_s": round(t_gen_base / n_q, 3),
        })

    # Persiste tempos (acumulando com execuções anteriores se já existir)
    tempos_json_path = base_paths["tempos_json"]
    tempos_csv_path = base_paths["tempos_csv"]
    prev_timings = []
    if tempos_json_path.exists():
        try:
            prev_timings = common.load_json(tempos_json_path)
        except Exception:
            prev_timings = []
    new_keys = {(r["strategy"], r["top_k"], r["semantic_percentile"], r["chunk_size"]) for r in timing_rows}
    merged_timings = [
        r for r in prev_timings
        if (r.get("strategy"), r.get("top_k"), r.get("semantic_percentile"), r.get("chunk_size")) not in new_keys
    ] + timing_rows
    common.save_json(merged_timings, tempos_json_path)
    if merged_timings:
        fieldnames = list(merged_timings[0].keys())
        with open(tempos_csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(merged_timings)
        print(f"\nTempos salvos em: {tempos_csv_path.absolute()}")

    print("\nFase 1 concluída. Encerre este processo antes da Fase 2 (VRAM limpa).")


if __name__ == "__main__":
    main()
