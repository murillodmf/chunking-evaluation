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
    "cartilha": {
        "pdf": "Textos_exemplo/cartilha_manejo_cigarrinhas.pdf",
        "qa": "Textos_exemplo/qa_cartilha_curado.json",  # curado a mao (15 pares);
        # Fase 0 (Qwen) desativada p/ este doc: QA gerado por LLM nao e gabarito
        "start_page": 0,          # cartilha Embrapa/IDR-PR: lê desde a primeira página
        "start_marker": None,     # sem recorte inicial
        "ref_markers": ["REFERÊNCIAS", "Referências", "REFERENCIAS", "Referencias",
                        "BIBLIOGRAFIA", "Bibliografia"],
        "drop_substrings": [],
        "area": "milho — manejo da cigarrinha e protocolos de diagnose (Cartilha Embrapa/IDR-PR)",
        "qa_example": (
            "Como funciona o método do saco plástico para amostragem da cigarrinha "
            "(amostragem, periodicidade e armazenamento)?"
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
# Corpus do benchmark final (10 PDFs em Documentos/Dataset/)
# ----------------------------------------------------------------------------
CORPUS_QA = "Textos_exemplo/qa_benchmark_3niveis.json"

_DEFAULT_REF_MARKERS = [
    "REFERÊNCIAS", "Referências", "REFERENCIAS", "Referencias",
    "BIBLIOGRAFIA", "Bibliografia",
]


def _corpus_doc(pdf: str, area: str, **overrides) -> dict:
    cfg = {
        "pdf": pdf,
        "start_page": 0,
        "start_marker": None,
        "ref_markers": _DEFAULT_REF_MARKERS,
        "drop_substrings": [],
        "block_chars": 5000,
        "area": area,
    }
    cfg.update(overrides)
    return cfg


CORPUS_DOCS = {
    "milho": _corpus_doc(
        "Documentos/Dataset/resistencia-de-hibridos-aos-enfezamentos.pdf",
        "milho — resistência de híbridos e severidade dos enfezamentos",
    ),
    "cartilha": _corpus_doc(
        "Documentos/Dataset/cartilha-manejo-das-cigarrinhas-e-enfezamentos.pdf",
        "milho — amostragem, diagnose e avaliação de enfezamentos",
    ),
    "oliveira-2007": _corpus_doc(
        "Documentos/Dataset/controle-quimico-da-cigarrinha-e-enfezamentos-por-molicutes.pdf",
        "milho — controle químico em viveiro e campo",
    ),
    "martins-2008": _corpus_doc(
        "Documentos/Dataset/eficiencia-de-inseticidas-contra-dalbulus-maidis.pdf",
        "milho — eficiência de inseticidas e amostragem",
    ),
    "xavier-2023": _corpus_doc(
        "Documentos/Dataset/tratamentos-para-controle-da-cigarrinha-revisao.pdf",
        "milho — revisão de tratamentos e controle biológico",
    ),
    "charles-2020": _corpus_doc(
        "Documentos/Dataset/guia-de-boas-praticas-para-o-manejo-da-cigarrinha-e-dos-enfezamentos.pdf",
        "milho — guia de boas práticas e manejo integrado",
    ),
    "embrapa-2022": _corpus_doc(
        "Documentos/Dataset/embrapa-2022-desafios-ao-manejo-de-enfezamentos-e-viroses-na-cultura-do-milho.pdf",
        "milho — transmissão, perdas regionais e manejo integrado",
    ),
    "revisao-2025": _corpus_doc(
        "Documentos/Dataset/artigo-revisao-2025-controle-biologico-de-dalbulus-maidis-com-macro-e-microrganismos.pdf",
        "milho — controle biológico com macro e microrganismos",
    ),
    "ufv-monitoramento-2023": _corpus_doc(
        "Documentos/Dataset/dissertacao-mestrado-ufv-2023-monitoramento-populacional-de-dalbulus-maidis-com-armadilhas-amarelas-adesivas.pdf",
        "milho — monitoramento com armadilhas amarelas adesivas",
    ),
    "ufv-incidencia-2023": _corpus_doc(
        "Documentos/Dataset/dissertacao-mestrado-ufv-2023-incidencia-de-dalbulus-maidis-e-enfezamentos-em-milho-silagem-em-goias-e-minas-gerais.pdf",
        "milho — incidência de enfezamentos e produtividade de silagem",
    ),
    "ufscar-2025": _corpus_doc(
        "Documentos/Dataset/tcc-ufscar-2025-efeito-residual-de-inseticidas-em-tratamento-de-sementes-no-controle-de-dalbulus-maidis.pdf",
        "milho — efeito residual curto de tratamento de sementes contra Dalbulus maidis (2025)",
    ),
    "revisao-genotipos-2025": _corpus_doc(
        "Documentos/Dataset/artigo-revisao-2025-estudos-recentes-sobre-resistencia-de-genotipos-de-milho-a-cigarrinha.pdf",
        "milho — estudos recentes de resistência genética, DC-EPG e dinâmica populacional (2025)",
    ),
    "ueopr-2025": _corpus_doc(
        "Documentos/Dataset/dissertacao-mestrado-ueopr-2025-comportamento-de-genotipos-de-milho-as-doencas-transmitidas-pelo-inseto-vetor.pdf",
        "milho — genótipos, resistência a inseticidas, amostragem, custos e bordaduras (2025)",
    ),
}

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
SEMANTIC_PERCENTILE = 60.0  # oficial (melhor answer_* no milho); p50/p65/p70 como ablacao
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

EMBEDDINGS_DEVICE = "auto"  # 'cuda' em GPUs >= 20GB (A100/L4) e 'cpu' em T4 (15GB)
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall",
                 "answer_correctness"]  # answer_correctness usa answer+ground_truth:
# unica alem de answer_relevancy que avalia o braço baseline (sem contexts)
RAGAS_TIMEOUT = 600
RAGAS_WORKERS = 1

SEED = 42

ANALISES_DIR = BASE_DIR / "analises"


def _k_suffix(k: int | None = None) -> str:
    """Retorna sufixo '_k5', '_k7', etc. quando k != TOP_K (3), preservando nomes originais para k=3."""
    if k is None or k == TOP_K:
        return ""
    return f"_k{k}"


def paths(doc: str, k: int | None = None) -> dict:
    """Resolve os caminhos de artefatos para um documento."""
    cfg = DOCS[doc]
    ks = _k_suffix(k)
    return {
        "pdf": BASE_DIR / cfg["pdf"],
        "qa": BASE_DIR / cfg["qa"],
        "geracoes": lambda strategy, _ks=ks: (
            ANALISES_DIR / f"geracoes_{doc}_{strategy}.json"
            if strategy == "baseline"
            else ANALISES_DIR / f"geracoes_{doc}{_ks}_{strategy}.json"
        ),
        "ragas_csv": ANALISES_DIR / f"ragas_results_{doc}{ks}.csv",
        "ragas_summary": ANALISES_DIR / f"ragas_summary_{doc}{ks}.csv",
        "tempos_json": ANALISES_DIR / f"tempos_{doc}.json",
        "tempos_csv": ANALISES_DIR / f"tempos_{doc}.csv",
    }


def corpus_paths(k: int | None = None) -> dict:
    """Resolve os caminhos de artefatos do benchmark multi-documento."""
    ks = _k_suffix(k)
    return {
        "qa": BASE_DIR / CORPUS_QA,
        "geracoes": lambda strategy, _ks=ks: (
            ANALISES_DIR / f"geracoes_corpus_{strategy}.json"
            if strategy == "baseline"
            else ANALISES_DIR / f"geracoes_corpus{_ks}_{strategy}.json"
        ),
        "ragas_csv": ANALISES_DIR / f"ragas_results_corpus{ks}.csv",
        "ragas_summary": ANALISES_DIR / f"ragas_summary_corpus{ks}.csv",
        "tempos_json": ANALISES_DIR / "tempos_corpus.json",
        "tempos_csv": ANALISES_DIR / "tempos_corpus.csv",
    }
