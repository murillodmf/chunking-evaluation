# Curadoria-2 — o que é cada documento (avaliado tecnicamente)

Triagem: gate de extração + chunkability real (fixed 512/51, recursive 512/51, semantic BGE-M3 p60).
Custo medido: BGE-M3 em CPU ~1 s/sentença no embed_documents -> 457 sents ≈ 8 min de embedding
na Fase 1; 1000 sents ≈ 17 min; 1800 sents ≈ 30 min (evitar).

| arquivo (novo nome) | tipo | instituição / veículo | ano | chars/sents | F/R/S60 | porte | veredito |
|---|---|---|---|---|---|---|---|
| embrapa-2022-desafios-ao-manejo-de-enfezamentos-e-viroses-na-cultura-do-milho.pdf | **Documento Embrapa** (Documentos 149, Embrapa Agropecuária Oeste) | Embrapa | 2022 | 66k/1020 | 144/51/409 | longo (~17 min) | RECOMENDADO |
| artigo-revisao-2025-controle-biologico-de-dalbulus-maidis-com-macro-e-microrganismos.pdf | **Artigo de revisão de literatura** | Revista Foco v.18 n.7 | 2025 | 53k/473 | 116/37/190 | ok (~8 min) | RECOMENDADO |
| dissertacao-mestrado-ufv-2023-incidencia-de-dalbulus-maidis-e-enfezamentos-em-milho-silagem-em-goias-e-minas-gerais.pdf | **Dissertação de mestrado** (Mestrado Profissional em Defesa Sanitária Vegetal) | UFV | 2023 | 43k/479 | 94/31/192 | ideal | RECOMENDADO |
| dissertacao-mestrado-ufv-2023-monitoramento-populacional-de-dalbulus-maidis-com-armadilhas-amarelas-adesivas.pdf | **Dissertação de mestrado** (Mestrado Profissional) | UFV | 2023 | 47k/466 | 102/34/187 | ok | RECOMENDADO |
| informativo-tecnico-epagri-2021-o-panorama-da-cigarrinha-em-santa-catarina.pdf | **Informativo Técnico** (RAC 111) | Epagri/SC | 2021 | 20k/153 | 44/16/62 | ideal (mais barato) | RECOMENDADO |
| dissertacao-mestrado-ufpa-2025-manejo-de-dalbulus-maidis-com-inseticidas-quimicos-e-biologicos.pdf | **Dissertação de mestrado** | UFPA | 2025 | 143k/1145 | 310/142/459 | muito longo (~20 min) | RELEVANTE (combina com N2-04/N3-03) |
| dissertacao-mestrado-ufpr-2023-comportamento-de-genotipos-de-milho-frente-ao-complexo-do-enfezamento.pdf | **Dissertação de mestrado** (Pós-Grad em Biotecnologia, Palotina) | UFPR | 2023 | 72k/457 | 156/59/lento | muito longo | RELEVANTE (resistência, como o milho) |
| dissertacao-mestrado-ueopr-2025-comportamento-de-genotipos-de-milho-as-doencas-transmitidas-pelo-inseto-vetor.pdf | **Dissertação de mestrado** | UEOPR | 2025 | 97k/900 | 211/69/361 | muito longo | RELEVANTE (genótipos) |
| dissertacao-mestrado-ufu-2018-controle-de-dalbulus-maidis-com-pulverizacao-eletrostatica.pdf | **Dissertação de mestrado** (Fitotecnia) | UFU | 2018 | 61k/784 | 132/45/314 | longo | RELEVANTE (tecnologia de aplicação) |
| tcc-agronomia-unipampa-2022-cigarrinha-na-cultura-do-milho.pdf | **Trabalho de conclusão de curso** (Bacharel em Agronomia) | UNIPAMPA | 2022 | 62k/693 | 136/46/278 | longo | RELEVANTE (panorama) |
| tcc-ufscar-2025-efeito-residual-de-inseticidas-em-tratamento-de-sementes-no-controle-de-dalbulus-maidis.pdf | **Trabalho de conclusão de curso** | UFSCar | 2025 | 92k/1170 | 199/67/469 | muito longo | RELEVANTE (residual de TS = nossa N2-02) |
| tcc-especializacao-ufpr-problematica-da-cigarrinha-do-milho-e-seu-controle.pdf | **Trabalho de conclusão** (curso de Especialista) | UFPR | s/ data no PDF | 48k/542 | 105/36/216 | ok | RELEVANTE (panorama) |
| tcc-especializacao-ufvjmj-taticas-de-controle-da-cigarrinha-do-milho.pdf | **Trabalho de conclusão** (curso de Especialista) | UFVJMJ | s/ data no PDF | 67k/673 | 147/48/lento | longo | RELEVANTE (táticas) |
| dissertacao-mestrado-uepg-2023-controle-microbiano-com-fungos-entomopatogenicos-no-complexo-de-enfezamentos.pdf | **Dissertação de mestrado** (Agronomia) | UEPG | 2023 | 134k/1396 | 290/103/lento | muito longo | RELEVANTE (Beauveria/Cordyceps) mas caro |
| dissertacao-mestrado-uepg-2026-adjuvante-nanotecnologico-para-aumento-do-periodo-de-controle-de-inseticidas.pdf | **Dissertação de mestrado** | UEPG | 2026 | 198k/1814 | 431/143/lento | **evitar** (~30 min) | NÃO RECOMENDADO (custo) |
| capitulo-livro-embrapa-2017-dalbulus-maidis-em-doencas-em-milho.pdf | **Capítulo de livro** (cap. 31, "Doenças em milho: insetos vetores, molicutes e vírus") | Embrapa | 2017 | 16k/223 | 34/13/90 | curto | RESERVA (denso e autoritativo) |
| artigo-revisao-2025-estudos-recentes-sobre-resistencia-de-genotipos-de-milho-a-cigarrinha.pdf | **Artigo de revisão** | Revista Observatório de la Economía Latinoamericana | 2025 | 37k/437 | 81/28/175 | ok | verificar a qualidade editorial: periódico de economia publicando revisão agronômica |
| artigo-doses-de-nitrogenio-e-densidades-no-acumulo-de-massa-em-milho-para-silagem.pdf | **Artigo científico** | (Cesar Beal) | s/ ano | 41k/440 | 89/32/177 | ok | **FORA DO TEMA**: nitrogênio/silagem, sem Cigarrinha nem enfezamento |
| resumo-congresso-2024-paisagem-agricola-e-populacoes-de-dalbulus-maidis-plantas-abrigo.pdf | **Resumo de congresso** (CBE/CPAC 2024) | SBIO | 2024 | 2,7k/29 | 6/2/12 | - | **REPROVADO**: sem texto integral (tema: plantas-abrigo) |
| artigo-revisao-2023-tratamentos-para-controle-da-cigarrinha-do-milho.pdf | **Artigo de revisão** | Journal of Exact Sciences | 2023 | 35k/305 | 75/25/123 | ideal | **DUPLICATA** (já está no dataset) |
| cartilha-embrapa-manejo-da-cigarrinha-e-enfezamentos-na-cultura-do-milho.pdf | **Cartilha** | Embrapa | s/ ano | 32k/318 | 70/23/128 | ideal | **DUPLICATA** (já está no dataset) |

## Leitura
- **Tipo de material:** 8 dissertações de mestrado, 5 trabalhos de conclusão de curso,
  2 artigos de revisão, 1 documento Embrapa, 1 cartilha, 1 capítulo de livro,
  1 informativo técnico, 1 artigo científico, 1 resumo de congresso.
- **Recência:** 2018-2026, com 8 docs de 2022 em diante. Responde à crítica de
  atualidade da pós-doutoranda.
- **Idioma:** 100% português (o lote anterior tinha 11 papers em inglês/spanhol).
- **Custo:** manter ≤500 sentes. O semântico é o gargalo (1 s/sentença na CPU).
- **Duplicatas:** os 2 últimos já estão no dataset (revisão 2023 e cartilha).
