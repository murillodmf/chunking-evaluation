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
    """Carrega LLM causal 4 bits e devolve (pipeline, tokenizer, gen_kwargs).

    Correções p/ RAGAS + A100:
    - pipeline criado SEM args de geração (evita conflito
      `max_new_tokens x max_length=20` do warning do transformers).
    - `return_full_text=False`: antes o pipe devolvia prompt+resposta,
      o que poluía as respostas e quebrava o parser JSON do RAGAS.
    - `clean_up_tokenization_spaces=False`: True é destrutivo p/ BPE
      (Llama/Qwen) e corrompia o JSON.
    - `gen_kwargs` vai em `HuggingFacePipeline(model_kwargs=...)`, não no
      construtor do pipeline. Use `build_hf_llm()` para montar.

    do_sample=False = greedy (deterministico): recomendado p/ o juiz RAGAS,
    que precisa emitir JSON estrito — amostragem gera saídas fora do schema.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

    _require_cuda()
    print(f"Carregando tokenizer para {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Configurando quantização de 4 bits (bitsandbytes)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"Carregando modelo {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        trust_remote_code=True,
    )
    try:
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    # O generation_config.json dos Instruct traz max_length=20: com
    # max_new_tokens setado isso gera o warning "Both ... set" (benigno:
    # max_new_tokens vence). Fixa o teto aqui para o valor pedido e loga,
    # para diagnosticar se o limite real aplicado confere com o config.
    try:
        model.generation_config.max_new_tokens = max_new_tokens
    except Exception:
        pass
    print(f"[gen] {model_id}: max_new_tokens={max_new_tokens} do_sample={do_sample}")
    # Sem max_new_tokens / temperature aqui: só flags de formato.
    # Sem truncation=True: ele usava o max_length do tokenizer e recriava
    # o conflito com max_new_tokens.
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        return_full_text=False, clean_up_tokenization_spaces=False,
    )
    gen_kwargs: dict = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature
    # Garante que nenhum max_length vaze para o generate (causa do warning
    # "Both max_new_tokens (=512) and max_length (=20)...").
    gen_kwargs.pop("max_length", None)
    vram_log(f"após carregar {model_id}")
    return pipe, tokenizer, gen_kwargs


def build_hf_llm(pipe, tokenizer, gen_kwargs: dict, use_chat_template: bool = False):
    """Monta o LLM LangChain com os kwargs no lugar certo.

    - Fase 1 (Llama, prompt manual já formatado): use_chat_template=False.
    - Fase 2 (juiz RAGAS, prompts crus do ragas): use_chat_template=True,
      que embrulha cada prompt via `tokenizer.apply_chat_template`
      (Qwen/Llama-Instruct só segue JSON estrito com chat template).
    """
    from langchain_huggingface import HuggingFacePipeline as _BaseHF

    # Sanitiza: greedy não aceita temperature/top_p/top_k (warning
    # "generation flags are not valid"); max_length conflita com max_new_tokens.
    clean = dict(gen_kwargs or {})
    clean.pop("max_length", None)
    if not clean.get("do_sample", True):
        for k in ("temperature", "top_p", "top_k"):
            clean.pop(k, None)
    greedy = not clean.get("do_sample", True)
    _tok = tokenizer  # closure: NAO passar como field pydantic (extra_forbidden)

    class _SanitizedHF(_BaseHF):  # type: ignore
        def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
            kwargs.pop("max_length", None)
            if greedy:
                for k in ("temperature", "top_p", "top_k"):
                    kwargs.pop(k, None)
            return super()._generate(
                prompts, stop=stop, run_manager=run_manager, **kwargs)

    if not use_chat_template:
        return _SanitizedHF(pipeline=pipe, model_kwargs=clean)

    # Juiz: aplica chat_template + filtra kwargs invasores do RAGAS/LangChain.
    class ChatTemplateHFPipeline(_SanitizedHF):  # type: ignore
        def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
            kwargs.pop("max_length", None)
            if greedy:
                for k in ("temperature", "top_p", "top_k"):
                    kwargs.pop(k, None)
            chat_prompts = []
            for p in prompts:
                try:
                    chat_prompts.append(_tok.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False, add_generation_prompt=True,
                    ))
                except Exception:
                    chat_prompts.append(p)
            return super()._generate(
                chat_prompts, stop=stop, run_manager=run_manager, **kwargs)

    return ChatTemplateHFPipeline(pipeline=pipe, model_kwargs=clean)


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
