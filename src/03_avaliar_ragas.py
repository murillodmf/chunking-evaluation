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
    args = ap.parse_args()

    common.setup_reproducibility()
    paths = config.paths(args.doc)
    common.vram_log("início Fase 2 (esperado ~0 GiB em processo novo)")

    print("=" * 60)
    print(f" FASE 2: AVALIAÇÃO RAGAS — doc='{args.doc}'")
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
    from ragas.llms import LangchainLLMWrapper as LangchainLLM
    from ragas.embeddings import LangchainEmbeddingsWrapper as LangchainEmbeddings

    strategies = ["fixed", "recursive", "semantic"]
    dfs = {}
    for s in strategies:
        p = paths["geracoes"](s)
        if not p.exists():
            raise SystemExit(f"Falta {p.name}: rode a Fase 1 (02_gerar_respostas.py --doc {args.doc}) primeiro.")
        dfs[s] = pd.DataFrame(common.load_json(p))
        print(f"{s}: {len(dfs[s])} registros carregados de {p.name}.")

    pipe, _ = common.build_causal_llm(
        config.MODELS["juiz"], config.RAG_TOKENS, config.RAG_TEMP)
    from langchain_huggingface import HuggingFacePipeline
    evaluator_llm = LangchainLLM(langchain_llm=HuggingFacePipeline(pipeline=pipe))
    evaluator_embeddings = LangchainEmbeddings(embeddings=common.build_embeddings())

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
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
        out[s] = scored
        for m in config.RAGAS_METRICS:
            if m in res:
                print(f"  {m}: {res[m]:.4f}")

    all_scores = pd.concat(out.values(), ignore_index=True)
    all_scores.to_csv(paths["ragas_csv"], index=False, encoding="utf-8")
    summary = all_scores.groupby("strategy")[config.RAGAS_METRICS].mean()
    summary.to_csv(paths["ragas_summary"], encoding="utf-8")

    print("\n" + "=" * 50)
    print(" TABELA COMPARATIVA GERAL (MÉDIAS) ".center(50, "="))
    print("=" * 50)
    print(summary)
    print(f"\nCSV detalhado: {paths['ragas_csv'].absolute()}")
    print(f"CSV resumo:    {paths['ragas_summary'].absolute()}")


if __name__ == "__main__":
    main()
