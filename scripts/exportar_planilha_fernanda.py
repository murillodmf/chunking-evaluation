"""Exporta planilha (.xlsx e .csv) para avaliação humana (Judge Secundário — Fernanda).

Lê os arquivos geracoes_corpus*.json em analises/ (para K=3, K=5, K=7, etc.) e
monta uma planilha organizada com:
- ID da questão (N1-01 .. N3-10) e Nível (Básico / Intermediário / Específico)
- Pergunta e Gabarito Oficial (ground_truth + fontes)
- Respostas geradas por cada estratégia (Fixed, Recursive, Semantic, Baseline)
- Documentos recuperados no Top-K
- Colunas para avaliação da especialista (Nota 1 a 5 e Observações)

Uso:
    python scripts/exportar_planilha_fernanda.py --k 3
    python scripts/exportar_planilha_fernanda.py --k 5
"""
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(BASE_DIR / "src"))
import config


def _col_letter(idx: int) -> str:
    res = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        res = chr(65 + rem) + res
    return res


def _build_sheet_xml(rows: list[list]) -> bytes:
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        '<cols>'
        '<col min="1" max="2" width="14" customWidth="1"/>'
        '<col min="3" max="4" width="42" customWidth="1"/>'
        '<col min="5" max="5" width="28" customWidth="1"/>'
        '<col min="6" max="9" width="45" customWidth="1"/>'
        '<col min="10" max="17" width="18" customWidth="1"/>'
        '</cols>',
        "<sheetData>",
    ]
    for r_idx, row in enumerate(rows, 1):
        if not row:
            lines.append(f'<row r="{r_idx}"/>')
            continue
        is_header = r_idx == 1
        row_ht = ' ht="30" customHeight="1"' if is_header else ' ht="85" customHeight="1"'
        lines.append(f'<row r="{r_idx}"{row_ht}>')
        for c_idx, val in enumerate(row, 1):
            if val is None or val == "":
                continue
            ref = f"{_col_letter(c_idx)}{r_idx}"
            style = "1" if is_header else "2"
            s_val = escape(str(val))
            lines.append(f'<c r="{ref}" t="inlineStr" s="{style}"><is><t>{s_val}</t></is></c>')
        lines.append("</row>")
    lines.append("</sheetData></worksheet>")
    return "\n".join(lines).encode("utf-8")


def write_xlsx(path: Path, sheets: list[tuple[str, list[list]]]) -> None:
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="10"/><name val="Arial"/></font>
    <font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Arial"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E79"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border>
      <left style="thin"><color rgb="FFD9D9D9"/></left>
      <right style="thin"><color rgb="FFD9D9D9"/></right>
      <top style="thin"><color rgb="FFD9D9D9"/></top>
      <bottom style="thin"><color rgb="FFD9D9D9"/></bottom>
      <diagonal/>
    </border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="3">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1">
      <alignment vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1">
      <alignment vertical="top" wrapText="1"/>
    </xf>
  </cellXfs>
</styleSheet>"""

    workbook_sheets, workbook_rels, content_types = [], [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
    ], [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, (name, rows) in enumerate(sheets, 1):
            workbook_sheets.append(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
            workbook_rels.append(
                f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
            )
            content_types.append(
                f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _build_sheet_xml(rows))

        workbook_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{"".join(workbook_sheets)}</sheets></workbook>'
        )
        root_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>"
        )
        zf.writestr("[Content_Types].xml", ("".join(content_types) + "</Types>").encode("utf-8"))
        zf.writestr("_rels/.rels", root_rels.encode("utf-8"))
        zf.writestr("xl/workbook.xml", workbook_xml.encode("utf-8"))
        zf.writestr("xl/_rels/workbook.xml.rels", ("".join(workbook_rels) + "</Relationships>").encode("utf-8"))
        zf.writestr("xl/styles.xml", styles_xml.encode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Exporta respostas para avaliação humana (Fernanda).")
    ap.add_argument("--k-list", default="3,5,7", help="Valores de K a incluir como abas (se existirem em analises/)")
    args = ap.parse_args()

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    qa_bench = json.loads((BASE_DIR / config.CORPUS_QA).read_text(encoding="utf-8"))

    # Baseline (independente de K)
    base_path = config.corpus_paths()["geracoes"]("baseline")
    base_data = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else []

    sheets = []
    candidate_strats = ["fixed", "recursive", "semantic", "semantic_p40", "semantic_p50", "semantic_p70"]
    for k_val in k_list:
        c_paths = config.corpus_paths(k=k_val)
        strat_data = {}
        for s in candidate_strats:
            p = c_paths["geracoes"](s)
            if p.exists():
                strat_data[s] = json.loads(p.read_text(encoding="utf-8"))
        if not strat_data:
            continue

        present_strats = list(strat_data.keys())
        header = [
            "ID",
            "Nível",
            "Pergunta",
            "Gabarito Oficial (Referência)",
            "Fontes Esperadas",
        ]
        for s in present_strats:
            header.append(f"Resposta {s.upper()} (K={k_val})")
        header.append("Resposta BASELINE (Sem RAG)")
        for s in present_strats:
            header.append(f"Nota Especialista — {s.upper()} (1 a 5)")
        header.extend([
            "Nota Especialista — BASELINE (1 a 5)",
            "Observações da Especialista (Fernanda)",
        ])
        for s in present_strats:
            header.append(f"Docs Recuperados {s.upper()} (K={k_val})")

        rows = [header]

        for i, q in enumerate(qa_bench):
            ans_bas = base_data[i].get("answer", "") if i < len(base_data) else ""
            row = [
                q.get("id", f"Q{i+1:02d}"),
                q.get("nivel", ""),
                q.get("question", ""),
                q.get("ground_truth", ""),
                ", ".join(q.get("documentos_fonte", [])),
            ]
            for s in present_strats:
                s_list = strat_data.get(s, [])
                row.append(s_list[i].get("answer", "") if i < len(s_list) else "")
            row.append(ans_bas)
            for _ in range(len(present_strats) + 2):
                row.append("")
            for s in present_strats:
                s_list = strat_data.get(s, [])
                row.append(", ".join(s_list[i].get("retrieved_docs", [])) if i < len(s_list) else "")
            rows.append(row)

        sheets.append((f"Avaliacao_TOP_K_{k_val}", rows))

        # Salva também CSV individual por K
        csv_path = config.ANALISES_DIR / f"avaliacao_fernanda_k{k_val}.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)
        print(f"CSV salvo: {csv_path}")

    if sheets:
        xlsx_path = config.ANALISES_DIR / "Avaliacao_Especialista_Fernanda.xlsx"
        write_xlsx(xlsx_path, sheets)
        print(f"Planilha Excel para a Fernanda salva em: {xlsx_path}")
    else:
        print("Nenhum arquivo geracoes_corpus*.json encontrado em analises/ ainda.")


if __name__ == "__main__":
    main()
