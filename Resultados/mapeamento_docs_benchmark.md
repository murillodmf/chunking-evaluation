# Mapeamento doc-id -> arquivo (benchmark 3 niveis)

Base: `Documentos/Dataset/` (6 PDFs em PT). Chaves usadas no campo `docs` de
`Textos_exemplo/qa_benchmark_3niveis.json`.

| doc-id no JSON | arquivo em Documentos/Dataset/ | ano | papel no benchmark |
|---|---|---|---|
| `milho` | resistencia-de-hibridos-aos-enfezamentos.pdf | 2023 | hibridos, resistencia, clima, produtividade (15 QA separados) |
| `cartilha` | cartilha-manejo-das-cigarrinhas-e-enfezamentos.pdf | 2017 | protocolos, diagnose PCR, amostragem, rendimento (15 QA separados) |
| `oliveira-2007` | controle-quimico-da-cigarrinha-e-enfezamentos-por-molicutes.pdf | 2007 | viveiro x campo, TS em semente, eficacia |
| `martins-2008` | eficiencia-de-inseticidas-contra-dalbulus-maidis.pdf | 2008 | eficiencia por via/idade, residual, armadilha |
| `xavier-2023` | tratamentos-para-controle-da-cigarrinha-revisao.pdf | 2023 | revisao, biologicos, contradicoes, faixa VE-V8 |
| `charles-2020` | guia-de-boas-praticas-para-o-manejo-da-cigarrinha-e-dos-enfezamentos.pdf | 2020 | 10 medidas oficiais, Agrofit, latencia 17-28 d, janela 20-30 d |

Uso: `charles-2020` aparece em 23 das 30 perguntas (entrou como base das
recomendacoes vigentes); `xavier-2023` e `oliveira-2007` sustentam os casos
divergentes; `martins-2008` e `cartilha` completam metodos e protocolos.

Observacao metodologica: os 3 primeiros papers (2007, 2008, 2023) sao
resultados de ensaio em condicoes especificas. As respostas do benchmark
separam o que foi observado do que os autores interpretam e registram as
condicoes, para que o sistema RAG nao seja avaliado por acertar recomendacao
desatualizada.
