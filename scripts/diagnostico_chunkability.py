"""Diagnóstico de chunkability de PDFs para curadoria do dataset.

Mede, por documento:
  - qualidade da extração (páginas, chars, sentenças, parágrafos);
  - como o texto se comporta nas 3 estratégias do TCC (fixed/recursive/semantic);
  - se os cortes caem em fronteira de frase (quebra "limpa") ou no meio.

Uso:
    python scripts/diagnostico_chunkability.py Documentos/Dataset
    python scripts/diagnostico_chunkability.py arq1.pdf arq2.pdf --out analises/chunkability.json
    python scripts/diagnostico_chunkability.py Documentos/Dataset --semantic

Sem --semantic, roda só fixed+recursive (rápido, sem GPU). O semântico usa
BGE-M3 na CPU (~1 s/sentença), então reserve para os finalistas.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "implementacoes"))
sys.path.append(str(BASE_DIR / "src"))

from fixed_size_chunker import MeuFixedSizeChunkerTCC
from recursive_chunker import MeuRecursiveChunkerTCC

try:
    import config
    import common
except ImportError:
    config = common = None

SENT_RE = re.compile(r"(?<=[.!?])\s+")
END_OK = tuple(".!?:;%)]}\"'")


def _corpus_cfg(pdf: Path) -> dict:
    """Acha a config do corpus pelo nome do PDF."""
    for cfg in config.CORPUS_DOCS.values():
        if (config.BASE_DIR / cfg["pdf"]).name == pdf.name:
            return cfg
    raise KeyError(f"PDF fora do CORPUS_DOCS: {pdf.name}")


def _n_paginas(pdf: Path) -> int:
    import pymupdf

    doc = pymupdf.open(str(pdf))
    try:
        return len(doc)
    finally:
        doc.close()


def extrair_texto(pdf: Path, pipeline: bool = False) -> tuple[str, dict]:
    if pipeline:
        if config is None:
            raise RuntimeError("config/common indisponíveis para --pipeline")
        cfg = _corpus_cfg(pdf)
        texto = common.load_and_clean_pdf(str(pdf), cfg)
        return texto, {"paginas": _n_paginas(pdf), "chars_por_pagina": len(texto) / max(_n_paginas(pdf), 1)}

    import pymupdf

    doc = pymupdf.open(str(pdf))
    try:
        paginas = len(doc)
        partes = [(p.get_text("text") or "") for p in doc]
    finally:
        doc.close()
    texto = "\n".join(partes)
    texto = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", texto)
    texto = re.sub(r"Figura \d+\..*?\.(?=\s*\n|\s*$)", "", texto, flags=re.DOTALL)
    linhas = []
    for linha in texto.split("\n"):
        s = linha.strip()
        if not s or re.match(r"^\d+$", s):
            continue
        if s.lower().startswith(("foto:", "fotos:")):
            continue
        linhas.append(linha)
    texto = "\n".join(linhas)
    texto = re.sub(r"(\w+)\s*-\s*\n\s*(\w+)", r"\1\2", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    return texto, {"paginas": paginas, "chars_por_pagina": len(texto) / max(paginas, 1)}


def sentencas(texto: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(texto) if s.strip()]


def estat_chunks(chunks: list[str], sent_set: set[str]) -> dict:
    if not chunks:
        return {}
    tamanhos = sorted(len(c) for c in chunks)
    n = len(chunks)
    limpos = 0
    inicio_ok = 0
    fim_ok = 0
    para_dup = [c.strip() for c in chunks]
    for c in chunks:
        t = c.strip()
        if not t:
            continue
        primeiro = t.lstrip()[:1]
        if primeiro.isupper() or primeiro.isdigit() or primeiro in "\"'([-":
            inicio_ok += 1
        if t.endswith(END_OK):
            fim_ok += 1
        if t.endswith(END_OK) and (
            primeiro.isupper() or primeiro.isdigit() or primeiro in "\"'([-"
        ):
            limpos += 1
    return {
        "n": n,
        "chars_medio": round(sum(tamanhos) / n, 1),
        "chars_mediana": tamanhos[n // 2],
        "chars_min": tamanhos[0],
        "chars_max": tamanhos[-1],
        "pct_fragmento_lt80": round(100 * sum(1 for x in tamanhos if x < 80) / n, 1),
        "pct_inicio_ok": round(100 * inicio_ok / n, 1),
        "pct_fim_ok": round(100 * fim_ok / n, 1),
        "pct_quebra_limpa": round(100 * limpos / n, 1),
        "pct_duplicado": round(100 * (n - len(set(para_dup))) / n, 1),
        "chunks_repetidos_em_sentencas": sum(1 for c in chunks if c.strip() in sent_set),
    }


def avaliar(pdf: Path, semantic: bool, embeddings, pipeline: bool = False) -> dict:
    texto, meta = extrair_texto(pdf, pipeline=pipeline)
    sents = sentencas(texto)
    sent_set = set(sents)
    npara = len([b for b in re.split(r"\n\s*\n", texto) if b.strip()])
    res = {
        "arquivo": pdf.name,
        "caminho": str(pdf),
        "paginas": meta["paginas"],
        "chars": len(texto),
        "nsent": len(sents),
        "dots": texto.count("."),
        "npara": npara,
        "chars_por_pagina": round(meta["chars_por_pagina"], 1),
    }
    fixed = MeuFixedSizeChunkerTCC(chunk_size=512, chunk_overlap=51)
    recursive = MeuRecursiveChunkerTCC(chunk_size=512, chunk_overlap=51)
    res["fixed"] = estat_chunks(fixed.split_text(texto), sent_set)
    res["recursive"] = estat_chunks(recursive.split_text(texto), sent_set)
    if semantic:
        from semantic_chunker import MeuSemanticChunkerTCC

        sem = MeuSemanticChunkerTCC(embeddings=embeddings, percentile_threshold=60.0)
        res["semantic_p60"] = estat_chunks(sem.split_text(texto), sent_set)

    flags = []
    if meta["paginas"] > 1 and meta["chars_por_pagina"] < 250 and len(sents) < 100:
        flags.append("EXTRACAO_SUSPEITA:chars_por_pagina<250")
    if len(sents) < 50 or res["dots"] < 10:
        flags.append("EXTRACAO_DEGENERADA")
    elif len(sents) < 150:
        flags.append("POUCAS_SENTENCAS")
    if npara <= 1:
        flags.append("SEM_PARAGRAFOS")
    if len(texto) < 20000:
        flags.append("CURTO")
    if len(texto) > 80000 or len(sents) > 800:
        flags.append("LONGO")
    if res["fixed"].get("pct_quebra_limpa", 100) < 20:
        flags.append("FIXED_CORTA_FRASE")
    if res["recursive"].get("chars_medio", 9999) < 600:
        flags.append("RECURSIVE_FRAGMENTA")
    if semantic and res["semantic_p60"].get("chars_medio", 9999) < 150:
        flags.append("SEMANTIC_FRAGMENTA")
    res["flags"] = flags
    res["veredito"] = _veredito(res)
    return res


def _veredito(r: dict) -> str:
    f = r["flags"]
    if any(x.startswith("EXTRACAO") for x in f):
        return "REPROVADO (extração)"
    if "POUCAS_SENTENCAS" in f:
        return "REPROVADO (poucas sentenças)"
    if "CURTO" in f:
        return "ATENÇÃO (curto)"
    if "LONGO" in f:
        return "ATENÇÃO (longo/caro)"
    return "OK"


def coletar(caminhos: list[str]) -> list[Path]:
    pdfs: list[Path] = []
    for c in caminhos:
        p = Path(c)
        if p.is_dir():
            pdfs.extend(sorted(p.glob("*.pdf")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
    return pdfs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("caminhos", nargs="+")
    ap.add_argument("--out", default=str(BASE_DIR / "analises" / "chunkability_final.json"))
    ap.add_argument("--semantic", action="store_true")
    ap.add_argument("--pipeline", action="store_true",
                    help="Usa config.CORPUS_DOCS + common.load_and_clean_pdf (corte de referências)")
    args = ap.parse_args()

    pdfs = coletar(args.caminhos)
    if not pdfs:
        raise SystemExit("Nenhum PDF encontrado.")

    embeddings = None
    if args.semantic:
        from langchain_huggingface import HuggingFaceEmbeddings

        print("Carregando BGE-M3 (CPU)...", flush=True)
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"}
        )

    resultados = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}", flush=True)
        try:
            r = avaliar(pdf, args.semantic, embeddings, pipeline=args.pipeline)
        except Exception as e:
            r = {"arquivo": pdf.name, "caminho": str(pdf), "erro": repr(e),
                 "flags": ["ERRO"], "veredito": "REPROVADO (erro)"}
        resultados.append(r)
        print(f"    {r.get('chars','?')} chars | {r.get('nsent','?')} sents | "
              f"F/R = {r.get('fixed',{}).get('n','?')}/{r.get('recursive',{}).get('n','?')} | "
              f"{r['veredito']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo: {out}")

    print("\n| arquivo | págs | chars | sents | F/R | %quebra limpa R | veredito |")
    print("|---|---:|---:|---:|---|---:|---|")
    for r in resultados:
        f = r.get("fixed", {})
        rec = r.get("recursive", {})
        print(f"| {r['arquivo'][:55]} | {r.get('paginas','?')} | {r.get('chars','?')} | "
              f"{r.get('nsent','?')} | {f.get('n','?')}/{rec.get('n','?')} | "
              f"{rec.get('pct_quebra_limpa','?')} | {r['veredito']} |")


if __name__ == "__main__":
    main()
