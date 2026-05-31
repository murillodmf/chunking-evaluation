# Projeto de TCC: Avaliação Empírica de Estratégias de Chunking em RAG

## Objetivo do Prompt
Preciso que você gere os scripts em Python necessários para implementar e isolar **três estratégias de chunking** dentro de um pipeline RAG (Retrieval-Augmented Generation) controlado. Este código fará parte dos experimentos do meu Trabalho de Conclusão de Curso (TCC), onde a única variável que deve mudar é a estratégia de segmentação de texto (Delineamento Experimental Controlado).

O ambiente de execução será **Windows nativo** e o framework de orquestração base deve ser o **LangChain** (versão estável atual).

---

## Requisitos Técnicos do Pipeline Fixo

Para garantir que os testes sejam justos, os seguintes componentes do pipeline RAG devem ser **estáticos** (idênticos para todos os testes):
1. **Banco de Dados Vetorial:** Utilizar **ChromaDB** configurado em modo *in-memory* (para rodar liso no Windows sem exigir persistência complexa ou Docker).
2. **Modelo de Embedding:** Configurar uma classe genérica ou usar um modelo open-source leve (ex: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFaceEmbeddings ou similar) para gerar os vetores.
3. **Métricas de Avaliação:** O script deve estar preparado para integrar com o framework **RAGAS** (capturando as listas de textos processados para posterior cálculo de *Faithfulness*, *Answer Relevance*, *Context Relevance* e *Context Recall*).
4. **Encoding:** Todos os inputs de leitura de arquivos de texto de domínio agronômico e geral devem usar explicitamente `encoding="utf-8"` para evitar quebras de caracteres no Windows.

---

## Especificação das 3 Estratégias de Chunking

Você deve criar um script que aplique e retorne o texto particionado das seguintes formas:

### 1. Fixed-size Chunking (Tamanho Fixo)
* **Abordagem:** Divisão estrita e simples por contagem de caracteres.
* **Implementação:** Utilizar a classe oficial `CharacterTextSplitter` do pacote `langchain-text-splitters`.
* **Parâmetros:** Deve permitir passar `chunk_size` e `chunk_overlap` dinamicamente como variáveis configuráveis.

### 2. Recursive Character Text Splitting
* **Abordagem:** Divisão hierárquica baseada em regras naturais de escrita para preservar a coesão.
* **Implementação:** Utilizar a classe oficial `RecursiveCharacterTextSplitter` do pacote `langchain-text-splitters`.
* **Parâmetros:** Configuração padrão com alvo de 512 tokens/caracteres, utilizando a lista hierárquica nativa de separadores (`["\n\n", "\n", ".", " "]`).

### 3. Semantic / Embedding-based Chunking (Customizado)
* **Contexto Importante:** Como o LangChain reestruturou o ecossistema e moveu/descontinuou o `SemanticChunker` do core estável (tornando-o difícil de rastrear sem dependências experimentais), **você deve implementar esta estratégia do zero criando uma classe customizada** que herde de `BaseDocumentTransformer` ou `TextSplitter` do LangChain.
* **Lógica da Implementação Semântica Necessária:**
  1. Receber o texto bruto e dividi-lo em sentenças individuais usando expressões regulares simples (`re.split`).
  2. Passar essas sentenças pelo mesmo modelo de embedding fixo do pipeline para gerar seus respectivos vetores.
  3. Calcular a distância de cosseno (`1 - similarity`) entre os embeddings de frases consecutivas (vizinhas).
  4. Utilizar o `numpy` para definir um ponto de corte dinâmico (*breakpoint*) baseado em um **Percentil estatístico** configurável (ex: percentil 70) das distâncias calculadas.
  5. Agrupar as frases em blocos (chunks) comuns e realizar a quebra estrita toda vez que a distância entre duas frases vizinhas ultrapassar o limite do percentil determinado.

---

## Entregáveis Esperados no Código

1. **`chunkers.py`**: Arquivo contendo a definição da classe customizada do `MeuSemanticChunker` e funções utilitárias para instanciar os outros dois splitters do LangChain.
2. **`pipeline.py`**: Um script de teste básico que:
   * Carregue um texto de exemplo usando `encoding="utf-8"`.
   * Passe esse texto pelos 3 chunkers separadamente.
   * Indexe os pedaços gerados no ChromaDB *in-memory*.
   * Mostre o output estruturado (uma lista limpa de strings/documentos para cada estratégia) pronto para ser enviado ao framework de avaliação RAGAS.

Por favor, gere um código limpo, modular, documentado e livre de caminhos de arquivos rígidos (use caminhos relativos ou `pathlib`).