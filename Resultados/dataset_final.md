# Dataset final — decisão (corpus de 10 documentos)

Critério técnico: extração limpa (gate nsent/dots) + nsent suficiente para o
percentil semântico + sem colapso em fixed/recursive/semantic + cobertura das
lacunas do benchmark (resistência de híbridos, amostragem/diagnose,
monitoramento regional, controle biológico recente).

Contagens abaixo já usam o pipeline final (`src/common.load_and_clean_pdf`), com
corte de referências pela última ocorrência após 50% do texto. Fixed/recursive:
512 tokens, overlap 51. Semantic: BGE-M3, p60.

## Corpus final (10 PDFs em `Documentos/Dataset/`)

| doc-id | chars | sents | fixed | recursive | semantic p60 | QAs |
|---|---:|---:|---:|---:|---:|---:|
| `milho` | 20.819 | 139 | 14 | 15 | 56 | 5 |
| `cartilha` | 25.862 | 204 | 18 | 18 | 82 | 3 |
| `oliveira-2007` | 24.717 | 172 | 17 | 17 | 69 | 15 |
| `martins-2008` | 17.696 | 121 | 14 | 15 | 49 | 16 |
| `xavier-2023` | 27.113 | 160 | 18 | 19 | 65 | 11 |
| `charles-2020` | 13.397 | 70 | 10 | 10 | 29 | 15 |
| `embrapa-2022` | 47.050 | 263 | 32 | 33 | 106 | 19 |
| `revisao-2025` | 47.429 | 284 | 31 | 32 | 114 | 5 |
| `ufv-monitoramento-2023` | 39.954 | 226 | 27 | 28 | 91 | 3 |
| `ufv-incidencia-2023` | 36.518 | 291 | 25 | 25 | 117 | 4 |
| **total** | **300.555** | **1.930** | **206** | **212** | **778** | **30** |

## Papel de cada documento

- `milho`: resistência/tolerância de híbridos, severidade, densidade de plantio.
- `cartilha`: amostragem (saco plástico/visual), diagnose por PCR, incidência,
  severidade e rendimento.
- `oliveira-2007`: viveiro telado x campo, tratamento de sementes, interpretação.
- `martins-2008`: eficiência por via/idade, residual, armadilha cônica.
- `xavier-2023`: revisão de tratamentos, biológicos, contradições, VE-V8.
- `charles-2020`: guia de boas práticas, 10 medidas, Agrofit, latência, janela.
- `embrapa-2022`: transmissão, perdas regionais, manejo integrado e cultural.
- `revisao-2025`: controle biológico com fungos/nematoides e limitações de campo.
- `ufv-monitoramento-2023`: armadilhas amarelas, densidade por região/janela/estádio.
- `ufv-incidencia-2023`: incidência e produtividade de silagem em GO/MG.

## Benchmark

- 30 QAs: 10 por nível, 22 `multi`, 5 `unico`, 3 `contraditoria`.
- Todos os 10 documentos aparecem em pelo menos 3 QAs.
- `charles-2020` caiu de 23 para 17 aparições; `embrapa-2022` é a fonte de maior
  cobertura (20), seguida de `charles-2020` e `martins-2008` (17).
- Doses, produtos e intervalos são dados experimentais; a resposta remete à bula,
  ao Agrofit e ao registro vigente.
- Endossulfan é tratado como resultado histórico (produto proibido no Brasil).
- As QAs de manejo (N3-06 e N3-10) são análises de cenário com ressalvas, não
  prescrições universais.

## Fora do corpus

- Duplicata: `Avaliacao-de-hibridos` = milho.
- Extração degenerada: `Resistencia-Sete Lagoas`, `milho_enfezamento`,
  `Estrategias-contra`, `Protecao-antecipada`.
- Curtos: `DIAGNOSE` e `Impacto` (movidos para `Documentos/_Descartados`).
- Grandes/caros: Pozebon, Neves, Medrada, Luz, Moritz, Carpane, Oliveira_Escape,
  Is_resistance, Redoan.
- Idioma EN/ES e fora do tema: Costa, Silva, Valarezo, soja.
- `Epagri-2021` ficou como reserva (6 atuais + 5 recomendados = 11; entraram os 4
  de maior lacuna).

## Pipeline

- Fase 1: `python src/02_gerar_respostas.py --corpus` (fixed/recursive/semantic +
  baseline). `--index-only` valida extração/chunking sem embeddings/LLM.
- Fase 2: `python src/03_avaliar_ragas.py --corpus`.
- Artefatos: `analises/geracoes_corpus_<estrategia>.json`,
  `analises/ragas_results_corpus.csv`, `analises/ragas_summary_corpus.csv`.
- Cada contexto recuperado carrega `doc_id` em `retrieved_docs`, permitindo medir
  acerto multi-documento.
