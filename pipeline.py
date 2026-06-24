import os
import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from fixed_size_chunker import MeuFixedSizeChunkerTCC
from recursive_chunker import MeuRecursiveChunkerTCC
from semantic_chunker import MeuSemanticChunkerTCC

def load_sample_text(file_path: str) -> str:
    """
    Carrega o arquivo de texto de exemplo garantindo o encoding UTF-8.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de texto de exemplo não encontrado em: {path.absolute()}")
    
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def print_separator(title: str):
    print("\n" + "="*80)
    print(f" {title.upper()} ".center(80, "="))
    print("="*80)

def main():
    # 1. Carregar o texto de exemplo (caminho dinâmico relativo à localização do script)
    script_dir = Path(__file__).parent.resolve()
    txt_path = script_dir / "state_of_the_union.txt"
    print(f"Carregando texto de exemplo de: {txt_path}")
    text_content = load_sample_text(str(txt_path))
    
    # 2. Inicializar o modelo de Embedding fixo (leve e rápido)
    print("\nInicializando HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 3. Instanciar as 3 estratégias de chunking utilizando as subclasses do TCC
    # Configuração experimental alvo: chunk_size=512 tokens e chunk_overlap=51 tokens.
    
    # Estratégia 1: Fixed-Size
    fixed_splitter = MeuFixedSizeChunkerTCC(chunk_size=512, chunk_overlap=51)
    # Estratégia 2: Recursive Character
    recursive_splitter = MeuRecursiveChunkerTCC(chunk_size=512, chunk_overlap=51)
    # Estratégia 3: Semantic (Customizado) com Percentil 70
    semantic_splitter = MeuSemanticChunkerTCC(embeddings=embeddings, percentile_threshold=70.0)
    
    # 4. Executar o split nas 3 estratégias
    print("\nExecutando segmentações...")
    
    chunks_fixed = fixed_splitter.split_text(text_content)
    chunks_recursive = recursive_splitter.split_text(text_content)
    chunks_semantic = semantic_splitter.split_text(text_content)
    
    # Inicializar codificador para exibir contagem de tokens no log
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def get_token_count(txt: str) -> int:
        return len(enc.encode(txt))
        
    # Salvar todos os chunks gerados pelas 3 técnicas em um único arquivo para fácil inspeção
    saida_path = script_dir / "saida.txt"
    with open(saida_path, "w", encoding="utf-8") as f:
        # 1. Escrever chunks da técnica Fixed-size
        f.write("================================================================================\n")
        f.write(" FIXED SIZE CHUNKS:\n")
        f.write("================================================================================\n\n")
        for idx, chunk in enumerate(chunks_fixed):
            f.write(f"--- Chunk {idx+1} (Tokens: {get_token_count(chunk)}, Caracteres: {len(chunk)}) ---\n")
            f.write(chunk)
            f.write("\n\n")
            
        # 2. Escrever chunks da técnica Recursive
        f.write("\n================================================================================\n")
        f.write(" RECURSIVE CHUNKS:\n")
        f.write("================================================================================\n\n")
        for idx, chunk in enumerate(chunks_recursive):
            f.write(f"--- Chunk {idx+1} (Tokens: {get_token_count(chunk)}, Caracteres: {len(chunk)}) ---\n")
            f.write(chunk)
            f.write("\n\n")
            
        # 3. Escrever chunks da técnica Semantic
        f.write("\n================================================================================\n")
        f.write(" SEMANTIC CHUNKS:\n")
        f.write("================================================================================\n\n")
        for idx, chunk in enumerate(chunks_semantic):
            f.write(f"--- Chunk {idx+1} (Tokens: {get_token_count(chunk)}, Caracteres: {len(chunk)}) ---\n")
            f.write(chunk)
            f.write("\n\n")
            
    print(f"-> Todos os chunks salvos com sucesso em: {saida_path.name}")
    
    print_separator("1. Fixed-Size Chunking (Tamanho Fixo - baseada em Tokens)")
    print(f"Total de chunks gerados: {len(chunks_fixed)}")
    for idx, chunk in enumerate(chunks_fixed):
        print(f"\n[Chunk {idx+1}] (Tokens: {get_token_count(chunk)}, Caracteres: {len(chunk)})")
        print(f"-> {chunk}")
        
    print_separator("2. Recursive Character Text Splitting (baseada em Tokens)")
    print(f"Total de chunks gerados: {len(chunks_recursive)}")
    for idx, chunk in enumerate(chunks_recursive):
        print(f"\n[Chunk {idx+1}] (Tokens: {get_token_count(chunk)}, Caracteres: {len(chunk)})")
        print(f"-> {chunk}")
        
    print_separator("3. Semantic / Embedding-based Chunking (Customizado)")
    print(f"Total de chunks gerados: {len(chunks_semantic)}")
    for idx, chunk in enumerate(chunks_semantic):
        print(f"\n[Chunk {idx+1}] (Tokens: {get_token_count(chunk)}, Caracteres: {len(chunk)})")
        print(f"-> {chunk}")
        
    # 5. Indexação no ChromaDB in-memory para teste RAG controlado
    print_separator("Indexação no ChromaDB (In-Memory)")
    
    # Criar coleções isoladas para cada estratégia
    print("Indexando Fixed-size chunks...")
    docs_fixed = [Document(page_content=c, metadata={"strategy": "fixed"}) for c in chunks_fixed]
    db_fixed = Chroma.from_documents(docs_fixed, embeddings, collection_name="fixed_chunks")
    
    print("Indexando Recursive chunks...")
    docs_recursive = [Document(page_content=c, metadata={"strategy": "recursive"}) for c in chunks_recursive]
    db_recursive = Chroma.from_documents(docs_recursive, embeddings, collection_name="recursive_chunks")
    
    print("Indexando Semantic chunks...")
    docs_semantic = [Document(page_content=c, metadata={"strategy": "semantic"}) for c in chunks_semantic]
    db_semantic = Chroma.from_documents(docs_semantic, embeddings, collection_name="semantic_chunks")
    
    print("Indexação concluída com sucesso no banco de dados vetorial in-memory!")
    
    # 6. Simular uma query de teste no RAG para cada estratégia
    query = "What are the plans for Intel's semiconductor mega site in Ohio?"
    print_separator(f"Teste de Recuperação RAG - Query: '{query}'")
    
    print("\n[Fixed-size Chunks] Top 1 resultado recuperado:")
    res_fixed = db_fixed.similarity_search(query, k=1)
    if res_fixed:
        print(f"-> {res_fixed[0].page_content}")
        
    print("\n[Recursive Chunks] Top 1 resultado recuperado:")
    res_recursive = db_recursive.similarity_search(query, k=1)
    if res_recursive:
        print(f"-> {res_recursive[0].page_content}")
        
    print("\n[Semantic Chunks] Top 1 resultado recuperado:")
    res_semantic = db_semantic.similarity_search(query, k=1)
    if res_semantic:
        print(f"-> {res_semantic[0].page_content}")
        
    # 7. Estruturar outputs de forma limpa para posterior envio ao framework RAGAS
    print_separator("Outputs prontos para o RAGAS")
    
    # Para o RAGAS, costumamos enviar a lista de contextos processados/recuperados.
    # Exemplo de estrutura que pode ser facilmente convertida para um Dataset do Ragas
    ragas_dataset_input = {
        "dataset_structure": {
            "fixed_chunks": chunks_fixed,
            "recursive_chunks": chunks_recursive,
            "semantic_chunks": chunks_semantic
        },
        "ragas_evaluation_notes": (
            "Para rodar a avaliação final com Ragas, você deve recuperar estes contextos "
            "e combiná-los com as respostas do LLM (answer) e a resposta de referência (ground_truth) "
            "no formato: Dataset.from_dict({'question': [...], 'answer': [...], 'contexts': [...], 'ground_truth': [...]})"
        )
    }
    
    # Imprimir em formato JSON limpo
    print(json.dumps(ragas_dataset_input["dataset_structure"], indent=2, ensure_ascii=False))
    print(f"\nNota: {ragas_dataset_input['ragas_evaluation_notes']}")

if __name__ == "__main__":
    main()
