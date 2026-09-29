"""Fase 0 — gera o dataset de pares pergunta-resposta com o Qwen 2.5 7B.

Uso:
    python src/01_gerar_dataset_qa.py --doc milho
    python src/01_gerar_dataset_qa.py --doc soja --blocos 4000
Saída: Textos_exemplo/qa_dataset_<doc>.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))
import config
import common


def qa_prompt(block: str, doc_cfg: dict) -> str:
    return f"""<|im_start|>system
Você é um especialista em Agronomia e Inteligência Artificial.
Baseando-se exclusivamente no texto técnico fornecido abaixo, sobre {doc_cfg['area']}, crie de 3 a 5 perguntas técnicas de alta qualidade acompanhadas de suas respectivas respostas de referência (ground_truth).

Regras obrigatórias:
1. As perguntas devem ser extremamente específicas e detalhadas, típicas de provas ou consultas técnicas de agrônomos (ex: evitar perguntas genéricas; prefira algo como "{doc_cfg['qa_example']}").
2. A resposta de referência (ground_truth) deve ser detalhada, completa e conter todos os fatos e dados numéricos presentes no texto que respondam à pergunta.
3. Retorne a resposta estritamente formatada como um array JSON válido de objetos, sem blocos de código adicionais (como ```json) ou introduções. Cada objeto do array deve ter as chaves "question" e "ground_truth".<|im_end|>
<|im_start|>user
Texto de referência:
---
{block}
---

Gere o array JSON de perguntas e respostas:<|im_end|>
<|im_start|>assistant
"""


def clean_model_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    if t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fase 0: gera dataset QA (Qwen 2.5 7B).")
    ap.add_argument("--doc", default=config.DOC_ATIVO, choices=list(config.DOCS),
                    help="Documento ativo (default: %(default)s)")
    ap.add_argument("--blocos", type=int, default=None,
                    help="Tamanho dos blocos em chars (default: do config)")
    ap.add_argument("--qa-out", default=None, help="Arquivo de saída (default: Textos_exemplo/qa_dataset_<doc>.json)")
    args = ap.parse_args()

    common.setup_reproducibility()
    doc_cfg = config.DOCS[args.doc]
    block_chars = args.blocos or doc_cfg["block_chars"]
    qa_out = Path(args.qa_out) if args.qa_out else config.paths(args.doc)["qa"]

    print("=" * 60)
    print(f" FASE 0: DATASET QA — doc='{args.doc}' ({doc_cfg['area']})")
    print("=" * 60)

    qwen_pipe, _ = common.build_causal_llm(
        config.MODELS["qa_gerador"], config.GEN_QA_TOKENS, config.GEN_QA_TEMP)
    print("Modelo Qwen 2.5 7B carregado com sucesso!\n")

    text = common.load_and_clean_pdf(str(config.paths(args.doc)["pdf"]), doc_cfg)
    print(f"Texto carregado: {len(text)} caracteres.")
    if len(text) < 1000:
        raise SystemExit(
            "Texto extraído muito curto (<1000 chars). Verifique start_page e "
            "a extração do PDF antes de gastar GPU.")

    blocks = common.split_blocks(text, block_chars)
    print(f"Texto dividido em {len(blocks)} blocos.\n")

    all_qa = []
    for i, block in enumerate(blocks):
        print(f"Gerando perguntas para o bloco {i + 1}/{len(blocks)}...")
        try:
            res = qwen_pipe(qa_prompt(block, doc_cfg), return_full_text=False)
            pairs = json.loads(clean_model_json(res[0]["generated_text"]))
            print(f"-> Gerou {len(pairs)} pares.")
            all_qa.extend(pairs)
        except Exception as e:
            print(f"-> Falhou neste bloco ({e}). Seguindo para o próximo.")

    print(f"\nTotal: {len(all_qa)} pares de QA.")
    common.save_json(all_qa, qa_out)


if __name__ == "__main__":
    main()
