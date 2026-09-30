"""Funções compartilhadas pelos scripts do pipeline (evita código duplicado)."""
import gc
import json
import random
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent.resolve()
BASE_DIR = SRC_DIR.parent
sys.path.append(str(BASE_DIR / "implementacoes"))

from fixed_size_chunker import MeuFixedSizeChunkerTCC
from recursive_chunker import MeuRecursiveChunkerTCC
from semantic_chunker import MeuSemanticChunkerTCC

import config


# ----------------------------------------------------------------------------
# Ambiente / reprodutibilidade
# ----------------------------------------------------------------------------
def setup_reproducibility(seed: int = config.SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def vram_log(tag: str) -> None:
    """Imprime a VRAM ocupada (GB). Ajuda a diagnosticar OOM na banca/Colab."""
    try:
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.memory_allocated() / 1024 ** 3
            print(f"[VRAM {tag}] {gb:.2f} GiB ocupados")
        else:
            print(f"[VRAM {tag}] CUDA indisponível")
    except ImportError:
        print(f"[VRAM {tag}] torch não instalado")


def free_vram() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# PDF genérico (parametrizado por documento — ver config.DOCS)
# ----------------------------------------------------------------------------
def _extract_with_pymupdf(pdf_path: str, start_page: int = 0) -> str:
    """Extração primária via PyMuPDF (melhor com fontes sem ToUnicode)."""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    try:
        parts = []
        for i in range(start_page, len(doc)):
            parts.append(doc[i].get_text("text") or "")
        return "\n".join(parts)
    finally:
        doc.close()


def _extract_with_pypdf(pdf_path: str, start_page: int = 0) -> str:
    """Extração de fallback via pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    extracted = []
    for i in range(start_page, len(reader.pages)):
        t = reader.pages[i].extract_text()
        if t:
            extracted.append(t)
    return "\n".join(extracted)


def _text_quality(text: str) -> dict:
    """Gate de qualidade: detecta extração degenerada (glifos /0 /1, sem pontuação)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return {
        "chars": len(text),
        "dots": text.count("."),
        "nsent": len(sentences),
    }


def load_and_clean_pdf(pdf_path: str, doc_cfg: dict) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF não encontrado em: {path.absolute()}")
    print(f"Lendo PDF de: {path.absolute()}")
    start_page = doc_cfg.get("start_page", 0)

    # 1) Tenta PyMuPDF (primário); 2) fallback pypdf. Escolhe o de melhor qualidade.
    candidates = {}
    try:
        candidates["pymupdf"] = _extract_with_pymupdf(str(path), start_page)
    except Exception as e:
        print(f"[extração] pymupdf falhou ({e}); tentando pypdf...")
    try:
        candidates["pypdf"] = _extract_with_pypdf(str(path), start_page)
    except Exception as e:
        print(f"[extração] pypdf falhou ({e}).")

    if not candidates:
        raise RuntimeError(f"Nenhum extrator conseguiu ler {path.name}. PDF escaneado? Considere OCR.")

    scored = {k: _text_quality(v) for k, v in candidates.items()}
    for k, q in scored.items():
        print(f"[extração] {k}: chars={q['chars']} dots={q['dots']} nsent={q['nsent']}")
    # Critério: mais sentenças; desempate por mais pontos.
    best = max(scored, key=lambda k: (scored[k]["nsent"], scored[k]["dots"]))
    full = candidates[best]
    print(f"[extração] selecionado: {best}")
    # Higiene: PDFs com fonte quebrada vazam controles binários (\x00-\x1f)
    # que corrompem chunking/embeddings. Limpeza neutra (só remove lixo).
    full = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", full)
    q = scored[best]
    if q["nsent"] < 10 or q["dots"] == 0:
        raise RuntimeError(
            f"Extração degenerada em {path.name} ({best}: nsent={q['nsent']}, dots={q['dots']}). "
            "Fonte sem mapa ToUnicode. Troque o PDF ou rode OCR (ocrmypdf/tesseract).")

    full = re.sub(r'Figura \d+\..*?\.(?=\s*\n|\s*$)', '', full, flags=re.DOTALL)

    drops = [d.lower() for d in doc_cfg.get("drop_substrings", [])]
    kept = []
    for line in full.split("\n"):
        l = line.strip()
        if not l or re.match(r"^\d+$", l):
            continue
        low = l.lower()
        if any(d in low for d in drops):
            continue
        if low.startswith("foto:") or low.startswith("fotos:"):
            continue
        kept.append(line)
    full = "\n".join(kept)
    full = re.sub(r"(\w+)\s*-\s*\n\s*(\w+)", r"\1\2", full)

    start = doc_cfg.get("start_marker")
    if start:
        idx = full.find(start)
        if idx != -1:
            full = full[idx:]

    for ref in doc_cfg.get("ref_markers", []):
        idx = full.find(ref)
        if idx != -1:
            full = full[:idx]
            break
    return full


def split_blocks(text: str, block_chars: int = 5000) -> list:
    """Divide em blocos grandes (~5000 chars) sem cortar parágrafo no meio."""
    blocks, i = [], 0
    while i < len(text):
        end = min(i + block_chars, len(text))
        if end < len(text):
            nl = text.find("\n\n", end)
            if nl != -1 and nl < end + 1000:
                end = nl + 2
        blocks.append(text[i:end].strip())
        i = end
    return [b for b in blocks if b]


# ----------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------
def _require_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA (GPU) não disponível. Use um runtime com GPU (ex: T4 no Colab).")


def build_causal_llm(model_id: str, max_new_tokens: int, temperature: float,
                     do_sample: bool = True):
    """Carrega LLM causal 4 bits e devolve (pipeline, tokenizer).

    do_sample=False = greedy (deterministico): recomendado p/ o juiz RAGAS,
    que precisa emitir JSON estrito — amostragem gera saídas fora do schema.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

    _require_cuda()
    print(f"Carregando tokenizer para {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Configurando quantização de 4 bits (bitsandbytes)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"Carregando modelo {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto"
    )
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **gen_kwargs)
    vram_log(f"após carregar {model_id}")
    return pipe, tokenizer


def build_embeddings(device: str = config.EMBEDDINGS_DEVICE):
    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"Inicializando embeddings {config.MODELS['embeddings']} (device={device})...")
    return HuggingFaceEmbeddings(
        model_name=config.MODELS["embeddings"], model_kwargs={"device": device}
    )


def build_chunkers(embeddings) -> dict:
    return {
        "fixed": MeuFixedSizeChunkerTCC(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP),
        "recursive": MeuRecursiveChunkerTCC(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP),
        "semantic": MeuSemanticChunkerTCC(
            embeddings=embeddings, percentile_threshold=config.SEMANTIC_PERCENTILE),
    }


# ----------------------------------------------------------------------------
# Artefatos em disco (contrato entre as fases)
# ----------------------------------------------------------------------------
def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"Salvo: {path.absolute()}")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_sample(qa: list, amostra: str) -> list:
    amostra = amostra.strip().lower()
    if amostra == "todas":
        return qa
    if amostra.isdigit():
        n = min(int(amostra), len(qa))
        print(f"Rodando amostra de {n}/{len(qa)} perguntas.")
        return qa[:n]
    raise ValueError(f"--amostra inválido: {amostra!r} (use N ou 'todas')")
