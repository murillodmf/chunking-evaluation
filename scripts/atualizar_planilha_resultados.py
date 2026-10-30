"""Atualiza TCC - Resultados.xlsx com todos os 37 experimentos concluídos no vLLM.
"""
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import pandas as pd
from pathlib import Path

ANALISES_DIR = Path(r"C:\Faculdade\TCC\analises")
EXCEL_PATH = ANALISES_DIR / "TCC - Resultados.xlsx"

# 1. Carrega dados
sum_k3 = pd.read_csv(ANALISES_DIR / "ragas_summary_corpus.csv")
sum_k5 = pd.read_csv(ANALISES_DIR / "ragas_summary_corpus_k5.csv")
sum_k7 = pd.read_csv(ANALISES_DIR / "ragas_summary_corpus_k7.csv")
tempos = pd.read_csv(ANALISES_DIR / "tempos_corpus.csv")

# Mapa de descrições dos parâmetros para a Aba 2
PARAM_DESCRICOES = {
    "baseline": "Conhecimento paramétrico puro do Llama-3.1-8B (sem retrieval)",
    "fixed": "Tamanho fixo = 512 tokens | Overlap = 51 tokens (10%) | 329 chunks",
    "fixed_256": "Ablação tamanho = 256 tokens | Overlap = 25 tokens (10%) | 650 chunks",
    "fixed_1024": "Ablação tamanho = 1024 tokens | Overlap = 102 tokens (10%) | 167 chunks",
    "recursive": "Recursivo = 512 tokens | Overlap = 51 tokens | Separadores padrão | 341 chunks",
    "recursive_256": "Ablação tamanho = 256 tokens | Overlap = 25 tokens | 676 chunks",
    "recursive_1024": "Ablação tamanho = 1024 tokens | Overlap = 102 tokens | 173 chunks",
    "semantic": "Threshold = Percentil 60.0 (Oficial) | 1.244 chunks (~388 chars médios)",
    "semantic_p40": "Threshold = Percentil 40.0 (Mais fragmentado) | 1.857 chunks",
    "semantic_p50": "Threshold = Percentil 50.0 | 1.547 chunks",
    "semantic_p70": "Threshold = Percentil 70.0 (Blocos maiores) | 938 chunks",
    "semantic_256": "Ablação Teto = 256 tokens máx no Semântico (p60) | 1.493 chunks",
    "semantic_1024": "Ablação Teto = 1024 tokens máx no Semântico (p60) | 1.247 chunks",
}

NOMES_EXIBICAO = {
    "baseline": "Baseline (Sem RAG)",
    "fixed": "Fixed-Size (512)",
    "fixed_256": "Fixed-Size (256 - Ablação)",
    "fixed_1024": "Fixed-Size (1024 - Ablação)",
    "recursive": "Recursive (512)",
    "recursive_256": "Recursive (256 - Ablação)",
    "recursive_1024": "Recursive (1024 - Ablação)",
    "semantic": "Semantic (p60 - Oficial)",
    "semantic_p40": "Semantic (p40 - Ablação)",
    "semantic_p50": "Semantic (p50 - Ablação)",
    "semantic_p70": "Semantic (p70 - Ablação)",
    "semantic_256": "Semantic-256 (Teto Tokens)",
    "semantic_1024": "Semantic-1024 (Teto Tokens)",
}

# Ordem desejada para apresentação dos resultados
ORDEM_ESTRATEGIAS = [
    "baseline",
    "fixed", "recursive", "semantic",
    "semantic_p40", "semantic_p50", "semantic_p70",
    "fixed_256", "fixed_1024",
    "recursive_256", "recursive_1024",
    "semantic_256", "semantic_1024"
]

def get_tempo_info(strategy, k_val):
    sub = tempos[(tempos["strategy"] == strategy) & (tempos["top_k"] == k_val)]
    if len(sub) == 0:
        # tenta achar em qualquer k para chunking
        sub_c = tempos[tempos["strategy"] == strategy]
        t_chunk = sub_c["tempo_chunking_s"].iloc[0] if len(sub_c) > 0 else 0
        return t_chunk, 0, 0
    row = sub.iloc[0]
    return row["tempo_chunking_s"], row["tempo_busca_medio_ms"], row["tempo_geracao_llm_medio_s"]

# Monta lista de 37 linhas consolidadas
linhas_consolidadas = []

# K=3 (13 estratégias)
for s in ORDEM_ESTRATEGIAS:
    m_row = sum_k3[sum_k3["strategy"] == s]
    if len(m_row) > 0:
        r = m_row.iloc[0]
        t_chunk, t_busca, t_gen = get_tempo_info(s, 0 if s == "baseline" else 3)
        linhas_consolidadas.append({
            "nome": NOMES_EXIBICAO.get(s, s),
            "top_k": 0 if s == "baseline" else 3,
            "param": PARAM_DESCRICOES.get(s, ""),
            "faithfulness": None if pd.isna(r.get("faithfulness")) else float(r["faithfulness"]),
            "answer_relevancy": None if pd.isna(r.get("answer_relevancy")) else float(r["answer_relevancy"]),
            "context_precision": None if pd.isna(r.get("context_precision")) else float(r["context_precision"]),
            "context_recall": None if pd.isna(r.get("context_recall")) else float(r["context_recall"]),
            "answer_correctness": None if pd.isna(r.get("answer_correctness")) else float(r["answer_correctness"]),
            "t_chunk": round(float(t_chunk), 3),
            "t_busca": round(float(t_busca), 2),
            "t_gen": round(float(t_gen), 3),
            "strategy_code": s,
        })

# K=5 (12 estratégias - sem baseline)
for s in ORDEM_ESTRATEGIAS:
    if s == "baseline":
        continue
    m_row = sum_k5[sum_k5["strategy"] == s]
    if len(m_row) > 0:
        r = m_row.iloc[0]
        t_chunk, t_busca, t_gen = get_tempo_info(s, 5)
        linhas_consolidadas.append({
            "nome": NOMES_EXIBICAO.get(s, s),
            "top_k": 5,
            "param": PARAM_DESCRICOES.get(s, ""),
            "faithfulness": None if pd.isna(r.get("faithfulness")) else float(r["faithfulness"]),
            "answer_relevancy": None if pd.isna(r.get("answer_relevancy")) else float(r["answer_relevancy"]),
            "context_precision": None if pd.isna(r.get("context_precision")) else float(r["context_precision"]),
            "context_recall": None if pd.isna(r.get("context_recall")) else float(r["context_recall"]),
            "answer_correctness": None if pd.isna(r.get("answer_correctness")) else float(r["answer_correctness"]),
            "t_chunk": round(float(t_chunk), 3),
            "t_busca": round(float(t_busca), 2),
            "t_gen": round(float(t_gen), 3),
            "strategy_code": s,
        })

# K=7 (12 estratégias - sem baseline)
for s in ORDEM_ESTRATEGIAS:
    if s == "baseline":
        continue
    m_row = sum_k7[sum_k7["strategy"] == s]
    if len(m_row) > 0:
        r = m_row.iloc[0]
        t_chunk, t_busca, t_gen = get_tempo_info(s, 7)
        linhas_consolidadas.append({
            "nome": NOMES_EXIBICAO.get(s, s),
            "top_k": 7,
            "param": PARAM_DESCRICOES.get(s, ""),
            "faithfulness": None if pd.isna(r.get("faithfulness")) else float(r["faithfulness"]),
            "answer_relevancy": None if pd.isna(r.get("answer_relevancy")) else float(r["answer_relevancy"]),
            "context_precision": None if pd.isna(r.get("context_precision")) else float(r["context_precision"]),
            "context_recall": None if pd.isna(r.get("context_recall")) else float(r["context_recall"]),
            "answer_correctness": None if pd.isna(r.get("answer_correctness")) else float(r["answer_correctness"]),
            "t_chunk": round(float(t_chunk), 3),
            "t_busca": round(float(t_busca), 2),
            "t_gen": round(float(t_gen), 3),
            "strategy_code": s,
        })

print(f"Total de linhas consolidadas preparadas: {len(linhas_consolidadas)} (Esperado: 37)")

# Abre o Excel original
wb = openpyxl.load_workbook(EXCEL_PATH)

# Corrige títulos das abas
wb.worksheets[0].title = "Execuções"
wb.worksheets[1].title = "1_Corpus_13_PDFs"
wb.worksheets[2].title = "2_Metricas_30_QAs"
wb.worksheets[3].title = "3_Explicacao_Metricas"

# Estilos
thin_border = Border(
    left=Side(style='thin', color='D3D3D3'),
    right=Side(style='thin', color='D3D3D3'),
    top=Side(style='thin', color='D3D3D3'),
    bottom=Side(style='thin', color='D3D3D3')
)
font_regular = Font(name="Calibri", size=10)
font_bold = Font(name="Calibri", size=10, bold=True)
fill_k3 = PatternFill(start_color="F2F7FA", end_color="F2F7FA", fill_type="solid")
fill_k5 = PatternFill(start_color="F0FFF0", end_color="F0FFF0", fill_type="solid")
fill_k7 = PatternFill(start_color="FFF8DC", end_color="FFF8DC", fill_type="solid")
fill_white = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")

# -------------------------------------------------------------------------
# ATUALIZA ABA 2: 2_Metricas_30_QAs
# -------------------------------------------------------------------------
ws2 = wb.worksheets[2]

# Cabeçalhos limpos
ws2.cell(1, 1, "ABA 2 — RESULTADOS RAGAS CONSOLIDADOS DO BENCHMARK (30 PERGUNTAS, 13 PDFs) — K=3, 5, 7 E ABLAÇÕES")
headers_ws2 = [
    "Estratégia / Configuração", "Top-K", "Parâmetro / Regra de Quebra",
    "Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall", "Answer Correctness",
    "Tempo Chunking (s)", "Busca Média (ms)", "Geração Média (s)"
]
for col_i, h_val in enumerate(headers_ws2, start=1):
    ws2.cell(2, col_i, h_val)

# Limpa linhas antigas a partir da linha 3
for r in range(3, ws2.max_row + 1):
    for c in range(1, 12):
        ws2.cell(r, c).value = None

for idx, item in enumerate(linhas_consolidadas, start=3):
    ws2.cell(idx, 1, item["nome"])
    ws2.cell(idx, 2, item["top_k"])
    ws2.cell(idx, 3, item["param"])
    ws2.cell(idx, 4, item["faithfulness"] if item["faithfulness"] is not None else "—")
    ws2.cell(idx, 5, item["answer_relevancy"] if item["answer_relevancy"] is not None else "—")
    ws2.cell(idx, 6, item["context_precision"] if item["context_precision"] is not None else "—")
    ws2.cell(idx, 7, item["context_recall"] if item["context_recall"] is not None else "—")
    ws2.cell(idx, 8, item["answer_correctness"] if item["answer_correctness"] is not None else "—")
    ws2.cell(idx, 9, item["t_chunk"])
    ws2.cell(idx, 10, item["t_busca"])
    ws2.cell(idx, 11, item["t_gen"])

    # Formatação e cores
    fill_row = fill_k3 if item["top_k"] == 3 or item["top_k"] == 0 else (fill_k5 if item["top_k"] == 5 else fill_k7)
    for c in range(1, 12):
        cell = ws2.cell(idx, c)
        cell.font = font_regular
        cell.border = thin_border
        cell.fill = fill_row
        if c in [2]:
            cell.alignment = Alignment(horizontal="center")
        elif c in [4, 5, 6, 7, 8]:
            cell.alignment = Alignment(horizontal="right")
            if isinstance(cell.value, float):
                cell.number_format = "0.0000"
        elif c in [9, 10, 11]:
            cell.alignment = Alignment(horizontal="right")
            if isinstance(cell.value, float):
                cell.number_format = "0.000" if c != 10 else "0.00"

# -------------------------------------------------------------------------
# ATUALIZA ABA 0: Execuções
# -------------------------------------------------------------------------
ws0 = wb.worksheets[0]
ws0.cell(1, 1, "PAINEL DE CONTROLE DE EXECUÇÕES E RESULTADOS RAG")
ws0.cell(2, 1, "Status consolidado de experimentos concluídos, matriz de parâmetros e métricas RAGAS obtidas.")
ws0.cell(4, 1, "1. MATRIZ DE CONFIGURAÇÕES EXPERIMENTAIS (X = Concluído com Resultados)")

headers_matriz = [
    "Técnica de Chunking", "Tamanho do Chunk (Tokens)", "Parâmetro Específico",
    "Top-K = 3", "Top-K = 5", "Top-K = 7", "Status da Configuração"
]
for col_i, h_val in enumerate(headers_matriz, start=1):
    ws0.cell(5, col_i, h_val)

# Atualiza Matriz de Configurações (linhas 6 a 18)
# Colunas: 1=Técnica, 2=Tamanho, 3=Parâmetro, 4=Top-K 3, 5=Top-K 5, 6=Top-K 7, 7=Status
matriz_data = [
    ("Baseline (Sem RAG)", "N/A (Prompt direto)", "Conhecimento paramétrico", "X (K=0)", "—", "—", "Concluído"),
    ("Fixed-Size Chunker", "512 tokens (Padrão)", "Overlap = 51 tokens (10%)", "X", "X", "X", "Concluído (Curva K completa)"),
    ("Fixed-Size Chunker", "256 tokens (Ablação)", "Overlap = 25 tokens (10%)", "X", "X", "X", "Concluído (Curva K completa)"),
    ("Fixed-Size Chunker", "1024 tokens (Ablação)", "Overlap = 102 tokens (10%)", "X", "X", "X", "Concluído (Curva K completa)"),
    ("Recursive Chunker", "512 tokens (Padrão)", "Overlap = 51 tokens (10%)", "X", "X", "X", "Concluído (Curva K completa)"),
    ("Recursive Chunker", "256 tokens (Ablação)", "Overlap = 25 tokens (10%)", "X", "X", "X", "Concluído (Curva K completa)"),
    ("Recursive Chunker", "1024 tokens (Ablação)", "Overlap = 102 tokens (10%)", "X", "X", "X", "Concluído (Curva K completa)"),
    ("Semantic Chunker", "Sem limite estrito (~388 chars)", "Percentil 40 (p40)", "X", "X", "X", "Concluído (Curva K completa)"),
    ("Semantic Chunker", "Sem limite estrito (~312 chars)", "Percentil 50 (p50)", "X", "X", "X", "Concluído (Curva K completa)"),
    ("Semantic Chunker", "Sem limite estrito (~388 chars)", "Percentil 60 (Oficial)", "X", "X", "X", "Concluído (Curva K completa)"),
    ("Semantic Chunker", "Sem limite estrito (~515 chars)", "Percentil 70 (p70)", "X", "X", "X", "Concluído (Curva K completa)"),
    ("Semantic Chunker (Teto)", "256 tokens máx (Ablação)", "Percentil 60 + max_tokens=256", "X", "X", "X", "Concluído (Curva K completa)"),
    ("Semantic Chunker (Teto)", "1024 tokens máx (Ablação)", "Percentil 60 + max_tokens=1024", "X", "X", "X", "Concluído (Curva K completa)"),
]

for row_idx, item in enumerate(matriz_data, start=6):
    for col_idx, val in enumerate(item, start=1):
        cell = ws0.cell(row_idx, col_idx, val)
        cell.font = font_regular
        cell.border = thin_border
        if col_idx in [4, 5, 6]:
            cell.alignment = Alignment(horizontal="center")
            cell.font = font_bold
            cell.fill = PatternFill(start_color="E6FFE6", end_color="E6FFE6", fill_type="solid")

# Atualiza tabela detalhada na aba Execuções (a partir da linha 21)
ws0.cell(19, 1, "2. RESULTADOS DETALHADOS DAS AVALIAÇÕES RAGAS (37 EXPERIMENTOS)")
ws0.cell(20, 1, "Configuração / Estratégia")
ws0.cell(20, 2, "Top-K")
ws0.cell(20, 3, "Faithfulness")
ws0.cell(20, 4, "Answer Relevancy")
ws0.cell(20, 5, "Context Precision")
ws0.cell(20, 6, "Context Recall")
ws0.cell(20, 7, "Answer Correctness")
ws0.cell(20, 8, "Tempo Chunking (s)")

for r in range(21, ws0.max_row + 1):
    for c in range(1, 9):
        ws0.cell(r, c).value = None

for idx, item in enumerate(linhas_consolidadas, start=21):
    ws0.cell(idx, 1, item["nome"])
    ws0.cell(idx, 2, item["top_k"])
    ws0.cell(idx, 3, item["faithfulness"] if item["faithfulness"] is not None else "—")
    ws0.cell(idx, 4, item["answer_relevancy"] if item["answer_relevancy"] is not None else "—")
    ws0.cell(idx, 5, item["context_precision"] if item["context_precision"] is not None else "—")
    ws0.cell(idx, 6, item["context_recall"] if item["context_recall"] is not None else "—")
    ws0.cell(idx, 7, item["answer_correctness"] if item["answer_correctness"] is not None else "—")
    ws0.cell(idx, 8, item["t_chunk"])

    fill_row = fill_k3 if item["top_k"] == 3 or item["top_k"] == 0 else (fill_k5 if item["top_k"] == 5 else fill_k7)
    for c in range(1, 9):
        cell = ws0.cell(idx, c)
        cell.font = font_regular
        cell.border = thin_border
        cell.fill = fill_row
        if c == 2:
            cell.alignment = Alignment(horizontal="center")
        elif c in [3, 4, 5, 6, 7]:
            cell.alignment = Alignment(horizontal="right")
            if isinstance(cell.value, float):
                cell.number_format = "0.0000"
        elif c == 8:
            cell.alignment = Alignment(horizontal="right")
            if isinstance(cell.value, float):
                cell.number_format = "0.000"

wb.save(EXCEL_PATH)
print("Planilha TCC - Resultados.xlsx atualizada com sucesso com TODOS os 37 experimentos!")
