# Mapeamento doc-id -> arquivo (benchmark 3 niveis)

Base: `Documentos/Dataset/` (10 PDFs em PT). Chaves usadas no campo `docs` de
`Textos_exemplo/qa_benchmark_3niveis.json` (30 QAs, 10 por nivel).

| doc-id no JSON | arquivo em Documentos/Dataset/ | ano | papel no benchmark |
|---|---|---|---|
| `milho` | resistencia-de-hibridos-aos-enfezamentos.pdf | 2023 | hibridos, tolerancia, severidade, densidade de plantio |
| `cartilha` | cartilha-manejo-das-cigarrinhas-e-enfezamentos.pdf | 2017 | protocolos, diagnose PCR, amostragem, incidencia/severidade, rendimento |
| `oliveira-2007` | controle-quimico-da-cigarrinha-e-enfezamentos-por-molicutes.pdf | 2007 | viveiro x campo, TS em semente, eficacia, interpretacao |
| `martins-2008` | eficiencia-de-inseticidas-contra-dalbulus-maidis.pdf | 2008 | eficiencia por via/idade, residual, armadilha conica |
| `xavier-2023` | tratamentos-para-controle-da-cigarrinha-revisao.pdf | 2023 | revisao, biologicos, contradicoes, faixa VE-V8 |
| `charles-2020` | guia-de-boas-praticas-para-o-manejo-da-cigarrinha-e-dos-enfezamentos.pdf | 2020 | 10 medidas oficiais, Agrofit, latencia 17-28 d, janela 20-30 d |
| `embrapa-2022` | embrapa-2022-desafios-ao-manejo-de-enfezamentos-e-viroses-na-cultura-do-milho.pdf | 2022 | transmissao, perdas regionais, manejo integrado, manejo cultural |
| `revisao-2025` | artigo-revisao-2025-controle-biologico-de-dalbulus-maidis-com-macro-e-microrganismos.pdf | 2025 | controle biologico, fungos/nematoides, limitacoes de campo |
| `ufv-monitoramento-2023` | dissertacao-mestrado-ufv-2023-monitoramento-populacional-de-dalbulus-maidis-com-armadilhas-amarelas-adesivas.pdf | 2023 | armadilhas amarelas, densidade por regiao/janela/estadio |
| `ufv-incidencia-2023` | dissertacao-mestrado-ufv-2023-incidencia-de-dalbulus-maidis-e-enfezamentos-em-milho-silagem-em-goias-e-minas-gerais.pdf | 2023 | incidencia e produtividade de silagem em GO/MG |

## Uso dos documentos no benchmark

| doc-id | QAs em que aparece |
|---|---|
| `embrapa-2022` | 19 |
| `martins-2008` | 16 |
| `oliveira-2007` | 15 |
| `charles-2020` | 15 |
| `xavier-2023` | 11 |
| `milho` | 5 |
| `revisao-2025` | 5 |
| `ufv-incidencia-2023` | 4 |
| `cartilha` | 3 |
| `ufv-monitoramento-2023` | 3 |

Todos os 10 documentos aparecem em pelo menos 3 QAs. `charles-2020` caiu de
23 para 17 aparicoes; `embrapa-2022` assume o papel de fonte tecnica mais
recente e de maior cobertura. `milho`, `cartilha` e as duas dissertacoes da UFV
entraram para cobrir resistencia de hibridos, metodos de amostragem/diagnose,
monitoramento regional e incidencia/produtividade.

## Observacoes metodologicas

- Os papers de 2007, 2008 e 2023 sao resultados de ensaio em condicoes
  especificas. As respostas do benchmark separam o que foi observado do que os
  autores interpretam e registram as condicoes, para que o sistema RAG nao seja
  avaliado por acertar recomendacao desatualizada.
- Doses, produtos e intervalos sao apresentados como dados experimentais; a
  resposta remete a bula, ao Agrofit e ao registro vigente.
- O resultado com endossulfan e tratado como historico (produto proibido no
  Brasil), sem recomendacao de uso.
- Percentuais de perda e eficacia sao condicionais a incidencia, epoca de
  infeccao, hibrido, ambiente, regiao e safra; nao sao generalizados.
- As QAs de manejo (N3-06 e N3-10) sao analises de cenario com ressalvas, nao
  prescricoes universais.
