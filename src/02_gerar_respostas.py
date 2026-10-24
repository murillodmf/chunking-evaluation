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
import sys
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
    ap.add_argument("--estrategias", default="fixed,recursive,semantic",
                    help="Chunkers a indexar/rodar (default: %(default)s)")
    ap.add_argument("--index-only", action="store_true",
                    help="Só extrai e chunka, sem embeddings/LLM (valida o corpus)")
    ap.add_argument("--sem-baseline", action="store_true",
                    help="Pula o braço baseline (LLM puro, sem retrieval)")
    args = ap.parse_args()

    common.setup_reproducibility()
    if args.corpus:
        paths = config.corpus_paths()
        doc_label = "corpus"
    else:
        paths = config.paths(args.doc)
        doc_label = args.doc

    print("=" * 60)
    print(f" FASE 1: GERAÇÃO RAG — doc='{doc_label}'")
    print("=" * 60)

    qa = common.select_sample(common.load_json(paths["qa"]), args.amostra)
    print(f"Dataset QA: {len(qa)} perguntas.")

    if args.corpus:
        doc_filter = [d.strip() for d in args.docs.split(",") if d.strip()] or None
        texts = common.load_corpus_texts(doc_filter)
        print(f"Corpus: {len(texts)} documentos, {sum(len(t) for t in texts.values())} caracteres.")
    else:
        doc_cfg = config.DOCS[args.doc]
        text = common.load_and_clean_pdf(str(paths["pdf"]), doc_cfg)
        print(f"Texto: {len(text)} caracteres.")

    from langchain_core.documents import Document
    from langchain_chroma import Chroma

    strategy_names = [s.strip() for s in args.estrategias.split(",") if s.strip()]
    valid = {"fixed", "recursive", "semantic"}
    unknown = [s for s in strategy_names if s not in valid]
    if unknown:
        raise SystemExit(f"--estrategias desconhecidas: {unknown}. Use: {list(valid)}")

    if args.index_only:
        from fixed_size_chunker import MeuFixedSizeChunkerTCC
        from recursive_chunker import MeuRecursiveChunkerTCC
        from semantic_chunker import MeuSemanticChunkerTCC

        chunkers = {}
        if "fixed" in strategy_names:
            chunkers["fixed"] = MeuFixedSizeChunkerTCC(
                chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        if "recursive" in strategy_names:
            chunkers["recursive"] = MeuRecursiveChunkerTCC(
                chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        if "semantic" in strategy_names:
            chunkers["semantic"] = MeuSemanticChunkerTCC(
                embeddings=common.build_embeddings(),
                percentile_threshold=config.SEMANTIC_PERCENTILE)

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
    chunkers = common.build_chunkers(embeddings)

    from transformers import pipeline as _  # noqa: F401 (garante backend carregado)
    dbs = {}
    for name in strategy_names:
        splitter = chunkers[name]
        if args.corpus:
            docs, counts = [], {}
            for doc_id, doc_text in texts.items():
                chunks = splitter.split_text(doc_text)
                counts[doc_id] = len(chunks)
                docs.extend(
                    Document(page_content=c, metadata={"strategy": name, "doc_id": doc_id})
                    for c in chunks
                )
            detail = ", ".join(f"{d}={n}" for d, n in counts.items())
            print(f"-> {name.upper()}: {len(docs)} chunks ({detail}).")
            collection = f"db_corpus_{name}"
        else:
            chunks = splitter.split_text(text)
            print(f"-> {name.upper()}: {len(chunks)} chunks.")
            docs = [
                Document(page_content=c, metadata={"strategy": name, "doc_id": args.doc})
                for c in chunks
            ]
            collection = f"db_{args.doc}_{name}"
        dbs[name] = Chroma.from_documents(docs, embeddings, collection_name=collection)
    print("Indexação concluída.")

    pipe, tokenizer, gen_kwargs = common.build_causal_llm(
        config.MODELS["rag_gerador"], config.RAG_TOKENS, config.RAG_TEMP)
    llm = common.build_hf_llm(pipe, tokenizer, gen_kwargs, use_chat_template=False)

    for strategy, db in dbs.items():
        print(f"\nGerando respostas: {strategy.upper()}")
        records = []
        per_ctx = max(500, config.RAG_MAX_CONTEXT_CHARS // max(1, args.k))
        for i, item in enumerate(qa):
            ctx_docs = db.similarity_search(item["question"], k=args.k)
            # Trava anti-OOM: cada contexto e truncado; o prompt nunca explode
            # mesmo que um chunk gigante seja recuperado.
            contexts = [d.page_content[:per_ctx] for d in ctx_docs]
            retrieved_docs = [d.metadata.get("doc_id") for d in ctx_docs]
            context_text = "\n\n".join(contexts)
            try:
                ans = llm.invoke(rag_prompt(context_text, item["question"]))
            except Exception as e:
                print(f"  Erro na pergunta {i + 1}: {e}")
                ans = "Erro na geração da resposta."
            records.append({
                "question": item["question"], "contexts": contexts,
                "retrieved_docs": retrieved_docs,
                "answer": ans, "ground_truth": item["ground_truth"],
            })
            print(f"  Pergunta {i + 1}/{len(qa)} processada.")
        common.save_json(records, paths["geracoes"](strategy))

    # Braço baseline: mesmo LLM, mesmas perguntas, ZERO contexto recuperado.
    # Mede o conhecimento paramétrico puro (Tabela 2 do TCC: RAG x LLM).
    if not args.sem_baseline:
        print("\nGerando respostas: BASELINE (sem retrieval)")
        records = []
        for i, item in enumerate(qa):
            try:
                ans = llm.invoke(baseline_prompt(item["question"]))
            except Exception as e:
                print(f"  Erro na pergunta {i + 1}: {e}")
                ans = "Erro na geração da resposta."
            records.append({
                "question": item["question"], "contexts": [],
                "retrieved_docs": [],
                "answer": ans, "ground_truth": item["ground_truth"],
            })
            print(f"  Pergunta {i + 1}/{len(qa)} processada.")
        common.save_json(records, paths["geracoes"]("baseline"))

    print("\nFase 1 concluída. Encerre este processo antes da Fase 2 (VRAM limpa).")


if __name__ == "__main__":
    main()
