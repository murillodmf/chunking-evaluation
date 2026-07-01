import os
import json
from pathlib import Path
from typing import List, Dict, Any

# LangChain Imports
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Importando os chunkers customizados
import sys
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir.parent / "implementacoes"))
sys.path.append(str(script_dir))

from fixed_size_chunker import MeuFixedSizeChunkerTCC
from recursive_chunker import MeuRecursiveChunkerTCC
from semantic_chunker import MeuSemanticChunkerTCC

def load_pdf_text(file_path: str) -> str:
    """
    Carrega o texto do PDF a partir da página 6 (índice 5, correspondente a página 38 do documento),
    iniciando na seção 'Exigências climáticas', aplicando filtros de ruído e removendo as referências ao final.
    """
    import re
    from pypdf import PdfReader
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF não encontrado em: {path.absolute()}")
    
    print(f"Lendo PDF de: {path.absolute()}")
    reader = PdfReader(str(path))
    
    extracted_text = []
    # Começa na página 6 (índice 5)
    for page_num in range(5, len(reader.pages)):
        page_text = reader.pages[page_num].extract_text()
        if page_text:
            extracted_text.append(page_text)
            
    full_text = "\n".join(extracted_text)
    
    # 1. Remover legendas de figuras (Figura X. ...) que interrompem a leitura
    fig_pattern = r'Figura \d+\..*?\.(?=\s*\n|\s*$)'
    full_text = re.sub(fig_pattern, '', full_text, flags=re.DOTALL)
    
    # 2. Filtrar linhas de cabeçalho, rodapé, créditos de imagem e numerações de página
    cleaned_lines = []
    for line in full_text.split('\n'):
        l = line.strip()
        # Ignorar linhas vazias
        if not l:
            continue
        # Ignorar numeração de página isolada
        if re.match(r'^\d+$', l):
            continue
        # Ignorar cabeçalhos e rodapés recorrentes do livro
        if "sistemas de produ" in l.lower() or "tecnologias de produ" in l.lower():
            continue
        # Ignorar créditos de fotos e linhas quebradas de nomes de fotógrafos
        if l.lower().startswith("foto:") or l.lower().startswith("fotos:") or "bouças farias" in l.lower() or "balbinot junior" in l.lower() or "arrabal arias" in l.lower():
            continue
        # Ignorar marcadores de imagem isolados (como "a b" ou "a b c")
        if re.match(r'^[a-e](\s+[a-e])*$', l.lower()):
            continue
            
        cleaned_lines.append(line)
        
    full_text = "\n".join(cleaned_lines)
    
    # 3. Reunir palavras hifenizadas que foram quebradas no fim da linha
    full_text = re.sub(r'(\w+)\s*-\s*\n\s*(\w+)', r'\1\2', full_text)
    
    # 4. Encontrar o início da seção "Exigências climáticas"
    target_start = "Exigências climáticas"
    start_idx = full_text.find(target_start)
    if start_idx != -1:
        full_text = full_text[start_idx:]
    else:
        # Busca alternativa com caixa baixa/alta se necessário
        alt_idx = full_text.lower().find(target_start.lower())
        if alt_idx != -1:
            full_text = full_text[alt_idx:]
            
    # 5. Remover referências bibliográficas do final se presentes
    ref_target = "Referências"
    ref_idx = full_text.find(ref_target)
    if ref_idx != -1:
        full_text = full_text[:ref_idx]
        
    return full_text

def print_separator(title: str):
    print("\n" + "="*80)
    print(f" {title.upper()} ".center(80, "="))
    print("="*80)

def main():
    # 1. Carregar e filtrar o texto do PDF de Ecofisiologia da Soja (caminho dinâmico relativo à localização do script)
    script_dir = Path(__file__).parent.resolve()
    pdf_path = script_dir.parent / "Textos_exemplo" / "ecofisiologia_soja_embrapa.pdf"
    
    try:
        text_content = load_pdf_text(str(pdf_path))
        print(f"Texto carregado com sucesso! Total de caracteres extraídos: {len(text_content)}")
    except Exception as e:
        print(f"Erro ao carregar o PDF: {e}")
        return
    
    # 2. Inicializar o modelo de Embedding fixo (BGE-M3)
    import torch
    print("\nInicializando HuggingFaceEmbeddings (BAAI/bge-m3)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    # 3. Instanciar as 3 estratégias de chunking usando as subclasses do TCC
    # Configurando tamanhos maiores para lidar com parágrafos técnicos mais densos
    fixed_splitter = MeuFixedSizeChunkerTCC(chunk_size=400, chunk_overlap=40)
    recursive_splitter = MeuRecursiveChunkerTCC(chunk_size=512, chunk_overlap=50)
    semantic_splitter = MeuSemanticChunkerTCC(embeddings=embeddings, percentile_threshold=70.0)
    
    # 4. Executar as segmentações nas 3 estratégias
    print("\nExecutando segmentações de texto...")
    
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

    # Exibir resumo dos chunks gerados
    print_separator("Resumo dos Chunks Gerados")
    print(f"1. Fixed-Size Chunking:      {len(chunks_fixed)} chunks")
    print(f"2. Recursive Character Split: {len(chunks_recursive)} chunks")
    print(f"3. Semantic Chunking:         {len(chunks_semantic)} chunks")
    
    # Mostrar amostra dos 3 primeiros chunks de cada estratégia
    print_separator("Amostra: Fixed-Size Chunks (Top 2)")
    for idx, chunk in enumerate(chunks_fixed[:2]):
        print(f"\n[Chunk {idx+1}] (Tokens: {get_token_count(chunk)}, Caracteres: {len(chunk)})\n-> {chunk[:300]}...")
        
    print_separator("Amostra: Recursive Chunks (Top 2)")
    for idx, chunk in enumerate(chunks_recursive[:2]):
        print(f"\n[Chunk {idx+1}] (Tokens: {get_token_count(chunk)}, Caracteres: {len(chunk)})\n-> {chunk[:300]}...")
        
    print_separator("Amostra: Semantic Chunks (Top 2)")
    for idx, chunk in enumerate(chunks_semantic[:2]):
        print(f"\n[Chunk {idx+1}] (Tokens: {get_token_count(chunk)}, Caracteres: {len(chunk)})\n-> {chunk[:300]}...")
        
    # 5. Indexação no ChromaDB in-memory
    print_separator("Indexação no ChromaDB (In-Memory)")
    
    print("Indexando Fixed-size chunks...")
    docs_fixed = [Document(page_content=c, metadata={"strategy": "fixed"}) for c in chunks_fixed]
    db_fixed = Chroma.from_documents(docs_fixed, embeddings, collection_name="fixed_ecofis")
    
    print("Indexando Recursive chunks...")
    docs_recursive = [Document(page_content=c, metadata={"strategy": "recursive"}) for c in chunks_recursive]
    db_recursive = Chroma.from_documents(docs_recursive, embeddings, collection_name="recursive_ecofis")
    
    print("Indexando Semantic chunks...")
    docs_semantic = [Document(page_content=c, metadata={"strategy": "semantic"}) for c in chunks_semantic]
    db_semantic = Chroma.from_documents(docs_semantic, embeddings, collection_name="semantic_ecofis")
    
    print("Indexação concluída com sucesso no ChromaDB!")
    
    # 6. Simulação de Teste de Recuperação RAG com Query sobre Ecofisiologia da Soja
    query = "Quais são as exigências térmicas e de temperatura do solo para a germinação da soja?"
    print_separator(f"Teste de Recuperação RAG - Query: '{query}'")
    
    print("\n[Fixed-size Chunks] Top 1 resultado:")
    res_fixed = db_fixed.similarity_search(query, k=1)
    if res_fixed:
        print(f"-> {res_fixed[0].page_content}")
        
    print("\n[Recursive Chunks] Top 1 resultado:")
    res_recursive = db_recursive.similarity_search(query, k=1)
    if res_recursive:
        print(f"-> {res_recursive[0].page_content}")
        
    print("\n[Semantic Chunks] Top 1 resultado:")
    res_semantic = db_semantic.similarity_search(query, k=1)
    if res_semantic:
        print(f"-> {res_semantic[0].page_content}")
        
    # 7. Estruturar outputs prontos para o RAGAS
    print_separator("Outputs estruturados para o RAGAS")
    ragas_dataset_input = {
        "fixed_chunks": chunks_fixed,
        "recursive_chunks": chunks_recursive,
        "semantic_chunks": chunks_semantic
    }
    
    # Exibe amostra JSON da contagem e chaves
    print(json.dumps({k: f"Lista com {len(v)} chunks prontos para RAGAS" for k, v in ragas_dataset_input.items()}, indent=2))

if __name__ == "__main__":
    main()
