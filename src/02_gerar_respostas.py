"""Fase 1 — geração RAG com LLaMA 3.1 8B (1 estratégia de chunking por vez).

Roda em processo próprio: ao terminar, salva geracoes_<doc>_<estrategia>.json
em analises/ e o processo morre (VRAM 100% limpa para a Fase 2).

Uso:
    python src/02_gerar_respostas.py --doc milho --amostra 5
    python src/02_gerar_respostas.py --doc milho --amostra todas
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


def rag_prompt(context_text: str, question: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{RAG_SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>
Contexto:
{context_text}

Pergunta:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Fase 1: geração RAG (LLaMA 3.1 8B).")
    ap.add_argument("--doc", default=config.DOC_ATIVO, choices=list(config.DOCS))
    ap.add_argument("--amostra", default="todas",
                    help="N de perguntas ou 'todas' (default: %(default)s)")
    ap.add_argument("--k", type=int, default=config.TOP_K, help="Top-K recuperados")
    args = ap.parse_args()

    common.setup_reproducibility()
    paths = config.paths(args.doc)
    doc_cfg = config.DOCS[args.doc]

    print("=" * 60)
    print(f" FASE 1: GERAÇÃO RAG — doc='{args.doc}'")
    print("=" * 60)

    qa = common.select_sample(common.load_json(paths["qa"]), args.amostra)
    print(f"Dataset QA: {len(qa)} perguntas.")
    text = common.load_and_clean_pdf(str(paths["pdf"]), doc_cfg)
    print(f"Texto: {len(text)} caracteres.")

    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFacePipeline

    embeddings = common.build_embeddings()  # CPU por default (config)
    chunkers = common.build_chunkers(embeddings)

    from transformers import pipeline as _  # noqa: F401 (garante backend carregado)
    dbs = {}
    for name, splitter in chunkers.items():
        chunks = splitter.split_text(text)
        print(f"-> {name.upper()}: {len(chunks)} chunks.")
        docs = [Document(page_content=c, metadata={"strategy": name}) for c in chunks]
        dbs[name] = Chroma.from_documents(docs, embeddings, collection_name=f"db_{args.doc}_{name}")
    print("Indexação concluída.")

    pipe, _ = common.build_causal_llm(
        config.MODELS["rag_gerador"], config.RAG_TOKENS, config.RAG_TEMP)
    llm = HuggingFacePipeline(pipeline=pipe)

    for strategy, db in dbs.items():
        print(f"\nGerando respostas: {strategy.upper()}")
        records = []
        per_ctx = max(500, config.RAG_MAX_CONTEXT_CHARS // max(1, args.k))
        for i, item in enumerate(qa):
            ctx_docs = db.similarity_search(item["question"], k=args.k)
            # Trava anti-OOM: cada contexto e truncado; o prompt nunca explode
            # mesmo que um chunk gigante seja recuperado.
            contexts = [d.page_content[:per_ctx] for d in ctx_docs]
            context_text = "\n\n".join(contexts)
            try:
                ans = llm.invoke(rag_prompt(context_text, item["question"]))
            except Exception as e:
                print(f"  Erro na pergunta {i + 1}: {e}")
                ans = "Erro na geração da resposta."
            records.append({
                "question": item["question"], "contexts": contexts,
                "answer": ans, "ground_truth": item["ground_truth"],
            })
            print(f"  Pergunta {i + 1}/{len(qa)} processada.")
        common.save_json(records, paths["geracoes"](strategy))

    print("\nFase 1 concluída. Encerre este processo antes da Fase 2 (VRAM limpa).")


if __name__ == "__main__":
    main()
