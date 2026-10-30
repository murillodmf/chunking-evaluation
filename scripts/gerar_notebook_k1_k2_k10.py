"""Gera o notebook e o arquivo de texto com as células para K=1, K=2 e K=10.
"""
import json
from pathlib import Path

BASE_DIR = Path(r"C:\Faculdade\TCC")
STRATEGIES = [
    ("fixed", "fixed"),
    ("recursive", "recursive"),
    ("semantic", "semantic"),
    ("semantic_p40", "semantic_p40"),
    ("semantic_p50", "semantic_p50"),
    ("semantic_p70", "semantic_p70"),
    ("fixed_256", "fixed_256"),
    ("fixed_1024", "fixed_1024"),
    ("recursive_256", "recursive_256"),
    ("recursive_1024", "recursive_1024"),
    ("semantic_256", "semantic_256"),
    ("semantic_1024", "semantic_1024"),
]

cells_txt = []
nb_cells = []

def add_md(text):
    cells_txt.append(f"# [MARKDOWN]\n{text.strip()}\n\n" + "="*70 + "\n\n")
    nb_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()]
    })

def add_code(comment, code):
    full_code = f"# {comment}\n" + code.strip()
    cells_txt.append(f"{full_code}\n\n" + "="*70 + "\n\n")
    nb_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in full_code.splitlines()]
    })

# Cabeçalho
add_md("""# TCC — Experimentos Adicionais: Curva Estendida de TOP-K (K=1, K=2 e K=10)
- **Objetivo**: Avaliar os extremos da curva de Top-K:
  - **K=1**: Mínimo absoluto (1 chunk recuperado — testa o recall estrito de primeira posição).
  - **K=2**: Contexto super enxuto (2 chunks).
  - **K=10**: Contexto expandido (10 chunks — testa saturação de ruído e 'lost in the middle').
- **Juiz**: Qwen2.5-14B-Instruct-AWQ via **vLLM** (8 workers paralelos, ~17 min por estratégia).
- **Sem Google Drive**: Cada bloco possui um botão de download direto via HTML/base64.""")

# Célula 0: Liberar VRAM
add_code("0 — Liberar VRAM do vLLM antes da Fase 1 (se estiver rodando na mesma sessão)", """
!pkill -9 -f vllm 2>/dev/null || true
import torch, gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
!nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
""")

# Célula 1: Setup / Clone
add_code("1 — Setup / Atualização do Repositório", """
%cd /content
!test -d chunking-evaluation || git clone https://github.com/murillodmf/chunking-evaluation.git
%cd /content/chunking-evaluation
!git fetch origin && git reset --hard origin/main
!git log -1 --oneline
!pip install -q -U bitsandbytes transformers accelerate
!pip install -q -r requirements.txt
from huggingface_hub import login
# Insira seu token Hugging Face caso solicitado:
login()
""")

# Célula 2: Upload de backup anterior
add_code("2 — Envie o último backup_vllm_FINAL_COMPLETO.zip baixado", """
%cd /content/chunking-evaluation
from google.colab import files
up = files.upload()
for nome in up:
    !unzip -o -q "{nome}" -d /content/chunking-evaluation
!ls analises/ | head -40
""")

add_md("## Fase 1 — Geração de Respostas (Llama-3.1-8B) para K=1, K=2 e K=10")

# Geração 1
add_code("3 — Fase 1 (Geração): Fixed, Recursive e Semantic (p40, p50, p60, p70) em K=1, 2, 10", """
%cd /content/chunking-evaluation
!python src/02_gerar_respostas.py --corpus --amostra todas --estrategias fixed,recursive,semantic --percentile-list 40,50,60,70 --k-list 1,2,10 --sem-baseline
""")

# Geração 2
add_code("4 — Fase 1 (Geração): Fixed e Recursive 256/1024 em K=1, 2, 10", """
%cd /content/chunking-evaluation
!python src/02_gerar_respostas.py --corpus --amostra todas --estrategias fixed_256,fixed_1024,recursive_256,recursive_1024 --k-list 1,2,10 --sem-baseline
""")

# Geração 3
add_code("5 — Fase 1 (Geração): Semantic com Teto 256/1024 em K=1, 2, 10", """
%cd /content/chunking-evaluation
!python src/02_gerar_respostas.py --corpus --amostra todas --estrategias semantic_256,semantic_1024 --k-list 1,2,10 --sem-baseline
""")

# Download Fase 1
download_fase1_code = """%cd /content/chunking-evaluation
!zip -q -r /content/backup_respostas_k1_k2_k10.zip analises/geracoes_corpus*.json analises/tempos_corpus.*
import base64
from IPython.display import HTML, display
caminho = "/content/backup_respostas_k1_k2_k10.zip"
with open(caminho, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
nome = "backup_respostas_k1_k2_k10.zip"
html = f'''
<a download="{nome}" href="data:application/zip;base64,{b64}">
    <button style="padding:12px 24px; background-color:#28a745; color:white; border:none; border-radius:6px; font-size:16px; font-weight:bold; cursor:pointer;">
        📥 Clique aqui para baixar {nome}
    </button>
</a>
'''
display(HTML(html))"""
add_code("6 — BAIXAR backup de todas as respostas geradas (K=1, 2, 10)", download_fase1_code)

add_md("## Fase 2 — Inicialização do Servidor vLLM (Qwen2.5-14B-AWQ)")

# Instalação vLLM
add_code("7 — Instala vLLM + cliente OpenAI", """
!pip install -q vllm langchain-openai
""")

# Servidor vLLM
add_code("8 — Inicia servidor vLLM em segundo plano (~3-5 min)", """
import subprocess, time, requests
log = open('/content/vllm.log', 'w')
srv = subprocess.Popen(
    ['vllm', 'serve', 'Qwen/Qwen2.5-14B-Instruct-AWQ',
     '--served-model-name', 'qwen-juiz',
     '--max-model-len', '16384', '--gpu-memory-utilization', '0.78',
     '--port', '8000'],
    stdout=log, stderr=subprocess.STDOUT)
for i in range(120):
    try:
        if requests.get('http://localhost:8000/v1/models', timeout=2).ok:
            print(f'vLLM PRONTO em ~{i*5}s'); break
    except Exception:
        pass
    time.sleep(5)
else:
    print('vLLM nao subiu — veja o log:')
    !tail -n 20 /content/vllm.log
""")

# Config vLLM
add_code("9 — Configura RAGAS para usar o vLLM (8 chamadas em paralelo)", """
import os
os.environ['JUDGE_BACKEND'] = 'vllm'
os.environ['VLLM_URL'] = 'http://localhost:8000/v1'
os.environ['VLLM_MODEL'] = 'qwen-juiz'
os.environ['RAGAS_WORKERS'] = '8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print('Configuração vLLM ativa!')
""")

cell_counter = 10
for k_val in [1, 2, 10]:
    add_md(f"## Avaliação RAGAS — Bloco TOP_K = {k_val} (12 Estratégias)")
    for label, strat in STRATEGIES:
        add_code(f"{cell_counter} — AVALIAR K={k_val} | {label}", f"""
%cd /content/chunking-evaluation
!python src/03_avaliar_ragas.py --corpus --k {k_val} --strategy {strat} --skip-existing
!cat analises/ragas_summary_corpus_k{k_val}.csv 2>/dev/null || true
""")
        cell_counter += 1
    
    # Download parcial do bloco K
    zip_nome = f"backup_vllm_k{k_val}.zip"
    dl_code = f"""%cd /content/chunking-evaluation
!zip -q -r /content/{zip_nome} analises/
import base64
from IPython.display import HTML, display
caminho = "/content/{zip_nome}"
with open(caminho, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
nome = "{zip_nome}"
html = f'''
<a download="{{nome}}" href="data:application/zip;base64,{{b64}}">
    <button style="padding:12px 24px; background-color:#007bff; color:white; border:none; border-radius:6px; font-size:16px; font-weight:bold; cursor:pointer;">
        📥 Clique aqui para baixar {{nome}}
    </button>
</a>
'''
display(HTML(html))"""
    add_code(f"{cell_counter} — BAIXAR resultados do Bloco K={k_val}", dl_code)
    cell_counter += 1

# Célula Final: Consolidação Completa
add_md("## Download Final Consolidado (Todos os K: 1, 2, 3, 5, 7, 10)")
dl_final = """%cd /content/chunking-evaluation
!zip -q -r /content/backup_vllm_TODOS_OS_K_FINAL.zip analises/
import base64
from IPython.display import HTML, display
caminho = "/content/backup_vllm_TODOS_OS_K_FINAL.zip"
with open(caminho, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
nome = "backup_vllm_TODOS_OS_K_FINAL.zip"
html = f'''
<a download="{nome}" href="data:application/zip;base64,{b64}">
    <button style="padding:14px 28px; background-color:#28a745; color:white; border:none; border-radius:8px; font-size:18px; font-weight:bold; cursor:pointer;">
        📦 BAIXAR PACOTE COMPLETO FINAL (TODOS OS K)
    </button>
</a>
'''
display(HTML(html))"""
add_code(f"{cell_counter} — BAIXAR PACOTE MESTRE COMPLETO (K=1, 2, 3, 5, 7, 10)", dl_final)

# Salva arquivo TXT
txt_path = BASE_DIR / "notebooks" / "celulas_k1_k2_k10.txt"
txt_path.write_text("".join(cells_txt), encoding="utf-8")
print("Salvo arquivo TXT:", txt_path)

# Salva arquivo IPYNB
nb_dict = {
    "cells": nb_cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}
ipynb_path = BASE_DIR / "notebooks" / "tcc_k1_k2_k10.ipynb"
with open(ipynb_path, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1, ensure_ascii=False)
print("Salvo arquivo IPYNB:", ipynb_path)
print(f"Total de células no notebook: {len(nb_cells)}")
