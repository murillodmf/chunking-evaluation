"""Fase 2 — avaliação RAGAS com Qwen 2.5 7B (processo próprio, VRAM limpa).

Lê analises/geracoes_<doc>_<estrategia>.json (gerados pela Fase 1) e salva
analises/ragas_results_<doc>.csv + analises/ragas_summary_<doc>.csv.

Uso:
    python src/03_avaliar_ragas.py --doc milho
Pré-requisito: ter rodado a Fase 1 para o mesmo --doc.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))
import config
import common


def main() -> None:
    ap = argparse.ArgumentParser(description="Fase 2: avaliação RAGAS (Qwen 2.5 7B).")
    ap.add_argument("--doc", default=config.DOC_ATIVO, choices=list(config.DOCS))
    ap.add_argument("--corpus", action="store_true",
                    help="Avalia o benchmark multi-documento (geracoes_corpus_*)")
    ap.add_argument("--strategy", default="todas",
                    help="Avalia 1 estrategia ou lista separada por virgula (ex: semantic_p40,semantic_p50,semantic_p70). Default: %(default)s")
    ap.add_argument("--k", type=int, default=config.TOP_K,
                    help="Valor de Top-K usado na Fase 1 (default: %(default)s)")
    ap.add_argument("--metrics", default=",".join(config.RAGAS_METRICS),
                    help="Subset separado por virgula p/ smoke test. Default: todas")
    args = ap.parse_args()

    common.setup_reproducibility()
    paths = config.corpus_paths(k=args.k) if args.corpus else config.paths(args.doc, k=args.k)
    doc_label = "corpus" if args.corpus else args.doc
    common.vram_log("início Fase 2 (esperado ~0 GiB em processo novo)")

    print("=" * 60)
    print(f" FASE 2: AVALIAÇÃO RAGAS — doc='{doc_label}' | K={args.k}")
    print("=" * 60)

    # Shim de compatibilidade: Ragas importa ChatVertexAI por padrão.
    from unittest.mock import MagicMock
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        dummy = MagicMock()
        dummy.ChatVertexAI = MagicMock
        sys.modules["langchain_community.chat_models.vertexai"] = dummy

    import pandas as pd
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.metrics import answer_correctness  # dispensa contexts: serve ao baseline
    from ragas.llms import LangchainLLMWrapper as LangchainLLM
    from ragas.embeddings import LangchainEmbeddingsWrapper as LangchainEmbeddings

    strategies = ["fixed", "recursive", "semantic"]
    if args.strategy != "todas":
        strategies = [s.strip() for s in args.strategy.split(",") if s.strip()]
    dfs = {}
    for s in strategies:
        p = paths["geracoes"](s)
        if not p.exists():
            flag = "--corpus" if args.corpus else f"--doc {args.doc}"
            raise SystemExit(f"Falta {p.name}: rode a Fase 1 (02_gerar_respostas.py {flag}) primeiro.")
        dfs[s] = pd.DataFrame(common.load_json(p))
        print(f"{s}: {len(dfs[s])} registros carregados de {p.name}.")

    wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]
    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }
    # Baseline (contexts=[]) nao admite metricas de retrieval: filtra sozinho.
    if strategies == ["baseline"]:
        precisa_ctx = {"faithfulness", "context_precision", "context_recall"}
        fora = [m for m in wanted if m in precisa_ctx]
        if fora:
            print(f"[baseline] sem contexts: ignorando {fora}.")
        wanted = [m for m in wanted if m not in precisa_ctx]
        if not wanted:
            wanted = ["answer_relevancy", "answer_correctness"]
            print(f"[baseline] usando {wanted}.")
    unknown = [m for m in wanted if m not in metric_map]
    if unknown:
        raise SystemExit(f"--metrics desconhecidas: {unknown}. Use: {list(metric_map)}")
    metrics = [metric_map[m] for m in wanted]

    pipe, tokenizer, gen_kwargs = common.build_causal_llm(
        config.MODELS["juiz"], config.JUDGE_TOKENS, config.JUDGE_TEMP,
        do_sample=config.JUDGE_DO_SAMPLE)
    evaluator_llm = LangchainLLM(
        langchain_llm=common.build_hf_llm(
            pipe, tokenizer, gen_kwargs, use_chat_template=True))
    evaluator_embeddings = LangchainEmbeddings(embeddings=common.build_embeddings())

    metrics = [metric_map[m] for m in wanted]

    def persist(out: dict) -> "pd.DataFrame":
        """Salva fundindo com o CSV ja em disco: preserva colunas de metricas
        calculadas em rodadas anteriores da mesma estrategia."""
        merged = pd.concat(out.values(), ignore_index=True)
        if paths["ragas_csv"].exists():
            try:
                prev = pd.read_csv(paths["ragas_csv"])
                other_strats = prev[~prev["strategy"].isin(list(out))]
                same_strats = prev[prev["strategy"].isin(list(out))]
                if not same_strats.empty:
                    # Preserva colunas de metricas anteriores para a mesma estrategia (alinhando por indice/ordem)
                    for s in list(out):
                        s_prev = same_strats[same_strats["strategy"] == s].reset_index(drop=True)
                        s_curr = merged[merged["strategy"] == s].reset_index(drop=True)
                        if len(s_prev) == len(s_curr):
                            for col in s_prev.columns:
                                if col not in s_curr.columns or s_curr[col].isna().all():
                                    merged.loc[merged["strategy"] == s, col] = s_prev[col].values
                merged = pd.concat([other_strats, merged], ignore_index=True)
            except Exception as e:
                print(f"[aviso] nao fundiu CSV previo ({e}); salvando so o atual.")
        merged.to_csv(paths["ragas_csv"], index=False, encoding="utf-8")
        all_metrics = [m for m in config.RAGAS_METRICS if m in merged.columns]
        summary = merged.groupby("strategy")[all_metrics].mean()
        summary.to_csv(paths["ragas_summary"], encoding="utf-8")
        return merged, summary

    out = {}
    for s, df in dfs.items():
        print(f"\nCalculando RAGAS para: {s.upper()}...")
        res = evaluate(
            dataset=Dataset.from_pandas(df), metrics=metrics,
            llm=evaluator_llm, embeddings=evaluator_embeddings,
            run_config=RunConfig(timeout=config.RAGAS_TIMEOUT, max_workers=config.RAGAS_WORKERS),
        )
        scored = res.to_pandas()
        scored["strategy"] = s
        scored["top_k"] = 0 if s == "baseline" else args.k
        out[s] = scored
        # Media + n valido por metrica (jobs com OOM/parse-fail viram NaN)
        for m in wanted:
            if m in scored.columns:
                n_ok = int(scored[m].notna().sum())
                print(f"  {m}: {scored[m].mean():.4f} (n_valid={n_ok}/{len(scored)})")
            else:
                print(f"  {m}: sem coluna (jobs falharam — ver NaN)")
        # Salvamento incremental fundido: se a proxima estrategia estourar VRAM,
        # o progresso desta (e das anteriores em disco) nao se perde.
        persist(out)
        common.free_vram()

    _, summary = persist(out)

    print("\n" + "=" * 50)
    print(" TABELA COMPARATIVA GERAL (MÉDIAS) ".center(50, "="))
    print("=" * 50)
    print(summary)
    print(f"\nCSV detalhado: {paths['ragas_csv'].absolute()}")
    print(f"CSV resumo:    {paths['ragas_summary'].absolute()}")


if __name__ == "__main__":
    main()
