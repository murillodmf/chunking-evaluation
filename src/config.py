"""Configuração central do projeto.

Tudo que varia entre experimentos mora aqui — trocar de documento, de modelo
ou de parâmetro de chunking é editar este arquivo (1 lugar só).
"""
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()

# ----------------------------------------------------------------------------
# Documentos (ativo: milho — o mais curto da pasta Documentos/)
# ----------------------------------------------------------------------------
DOCS = {
    "milho": {
        "pdf": "Textos_exemplo/milho_hibridos_enfezamento.pdf",
        "qa": "Textos_exemplo/qa_milho_curado.json",  # curado a mao (15 pares);
        # Fase 0 (Qwen) desativada p/ este doc: QA gerado por LLM nao e gabarito
        "start_page": 0,          # artigo curto: lê desde a primeira página
        "start_marker": None,     # sem recorte inicial
        "ref_markers": ["REFERÊNCIAS", "Referências", "REFERENCIAS", "Referencias"],
        "drop_substrings": [],    # sem cabeçalhos específicos conhecidos
        "area": "milho — cigarrinha-do-milho e complexo de enfezamentos",
        "qa_example": (
            "Quais híbridos experimentais se destacaram pela resistência "
            "aos enfezamentos no ensaio de Sete Lagoas/MG?"
        ),
        "block_chars": 5000,
    },
    "soja": {  # legado: mantém as regras do experimento original
        "pdf": "Textos_exemplo/ecofisiologia_soja_embrapa.pdf",
        "qa": "Textos_exemplo/qa_dataset_soja.json",
        "start_page": 5,
        "start_marker": "Exigências climáticas",
        "ref_markers": ["Referências"],
        "drop_substrings": [
            "sistemas de produ", "tecnologias de produ",
            "bouças farias", "balbinot junior", "arrabal arias",
        ],
        "area": "soja — ecofisiologia (material Embrapa)",
        "qa_example": (
            "Qual o efeito de temperaturas do solo abaixo de 20°C "
            "na germinação da soja?"
        ),
        "block_chars": 5000,
    },
}

DOC_ATIVO = "milho"

# ----------------------------------------------------------------------------
# Modelos (100% locais, quantização 4 bits)
# ----------------------------------------------------------------------------
MODELS = {
    "qa_gerador": "Qwen/Qwen2.5-7B-Instruct",      # gera pares pergunta-resposta
    "rag_gerador": "meta-llama/Llama-3.1-8B-Instruct",  # responde (Fase 1)
    "juiz": "Qwen/Qwen2.5-14B-Instruct",  # avalia RAGAS (Fase 2). Upgrade piloto:
    # o 7B repetia o enunciado e errava o schema JSON do RAGAS (parse-fail
    # sistematico). 14B 4-bit ~= 8 GiB: folga total na A100 (40GB).
    "embeddings": "BAAI/bge-m3",
}

BNB_4BIT = dict(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",  # resolvido para torch.float16 em common.py
)

# ----------------------------------------------------------------------------
# Parâmetros do pipeline
# ----------------------------------------------------------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 51
SEMANTIC_PERCENTILE = 65.0  # exp p65 (p50=70chunks p60=56 p65=49 p70=43chunks, contagem local CPU)
# p70 gerava blocos de >10k tokens; p30 fragmentava em 1 frase/chunk.
# Percentil e parametro DO metodo (experimento continua puro); vale p/ todos os docs.
TOP_K = 3
RAG_MAX_CONTEXT_CHARS = 6000  # trava: prompt do Llama nunca passa disso em contexto

GEN_QA_TOKENS = 1024
GEN_QA_TEMP = 0.3
RAG_TOKENS = 512
RAG_TEMP = 0.1
JUDGE_TOKENS = 1024  # faithfulness gera 4-6 statements+reason+verdict; 256/512 truncava o JSON no meio (verdict missing).
# Na A100 (40GB) cabe; smoke na T4: prefira --metrics faithfulness.
JUDGE_TEMP = 0.1  # ignorado com greedy (mantido p/ documentacao)
JUDGE_DO_SAMPLE = False  # greedy: deterministico, melhor p/ JSON estrito do RAGAS

EMBEDDINGS_DEVICE = "cpu"  # BGE-M3 na CPU libera ~2,3 GB de VRAM p/ os LLMs
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall",
                 "answer_correctness"]  # answer_correctness usa answer+ground_truth:
# unica alem de answer_relevancy que avalia o braço baseline (sem contexts)
RAGAS_TIMEOUT = 600
RAGAS_WORKERS = 1

SEED = 42

ANALISES_DIR = BASE_DIR / "analises"


def paths(doc: str) -> dict:
    """Resolve os caminhos de artefatos para um documento."""
    cfg = DOCS[doc]
    return {
        "pdf": BASE_DIR / cfg["pdf"],
        "qa": BASE_DIR / cfg["qa"],
        "geracoes": lambda strategy: ANALISES_DIR / f"geracoes_{doc}_{strategy}.json",
        "ragas_csv": ANALISES_DIR / f"ragas_results_{doc}.csv",
        "ragas_summary": ANALISES_DIR / f"ragas_summary_{doc}.csv",
    }
