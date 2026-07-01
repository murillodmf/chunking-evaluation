import os
import json
import pandas as pd
from pathlib import Path
import gc
import torch

# Importando os components do LangChain
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configurações de import dos chunkers customizados
import sys
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir.parent / "implementacoes"))
sys.path.append(str(script_dir))

# Se o usuário rodar no repositório local
from fixed_size_chunker import MeuFixedSizeChunkerTCC
from recursive_chunker import MeuRecursiveChunkerTCC
from semantic_chunker import MeuSemanticChunkerTCC
from gerar_dataset_qa import load_and_clean_pdf

def load_llama_model():
    """
    Carrega o modelo LLaMA 3.1 8B Instruct localmente com quantização de 4 bits para a geração RAG.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
    from langchain_huggingface import HuggingFacePipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("CUDA (GPU) não disponível. Necessário GPU.")
        
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Carregando tokenizer para {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Configurando quantização de 4 bits...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Carregando modelo gerador {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    llama_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True
    )
    
    return HuggingFacePipeline(pipeline=llama_pipe)

def get_llama_response(prompt: str, llama_pipeline) -> str:
    """
    Gera a resposta do RAG utilizando o LLaMA 3.1 local.
    """
    try:
        response = llama_pipeline.invoke(prompt)
        if prompt in response:
            response = response.replace(prompt, "")
        return response.strip()
    except Exception as e:
        print(f"Erro ao gerar resposta com o LLaMA: {e}")
        return "Erro na geração da resposta."

def load_qwen_model():
    """
    Carrega o modelo Qwen 2.5 7B Instruct localmente com quantização de 4 bits para a avaliação RAGAS.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
    from langchain_huggingface import HuggingFacePipeline
    
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Carregando tokenizer para {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Configurando quantização de 4 bits...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Carregando modelo avaliador {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    qwen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True
    )
    
    return HuggingFacePipeline(pipeline=qwen_pipe)

def run_rag_evaluation():
    print("="*60)
    print(" AVALIAÇÃO RAG - 100% LOCAL (LLaMA 3.1 + Qwen 2.5 + BGE-M3)")
    print("="*60)
    
    # 1. Carregar o Dataset de QA
    qa_path = script_dir.parent / "Textos_exemplo" / "qa_dataset.json"
    if not qa_path.exists():
        qa_path = Path("Textos_exemplo/qa_dataset.json")
        
    if not qa_path.exists():
        print(f"Erro: Dataset de QA não encontrado em {qa_path.absolute()}. Execute 'gerar_dataset_qa.py' primeiro.")
        return
        
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
        
    print(f"Dataset de QA carregado. Total de perguntas: {len(qa_dataset)}")
    
    print("Deseja rodar para todas as perguntas ou apenas uma amostra de teste (ex: 5 perguntas)?")
    amostra_opt = input("Digite 'todas' ou o número de perguntas (ex: 5): ").strip().lower()
    if amostra_opt != "todas" and amostra_opt.isdigit():
        qa_dataset = qa_dataset[:int(amostra_opt)]
        print(f"Rodando teste para {len(qa_dataset)} perguntas.")

    # 2. Carregar e limpar texto do PDF
    pdf_path = script_dir.parent / "Textos_exemplo" / "ecofisiologia_soja_embrapa.pdf"
    if not pdf_path.exists():
        pdf_path = Path("Textos_exemplo/ecofisiologia_soja_embrapa.pdf")
        
    text_content = load_and_clean_pdf(str(pdf_path))
    
    # 3. Inicializar modelo de Embedding (BGE-M3)
    print("\nInicializando HuggingFaceEmbeddings (BAAI/bge-m3)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    # 4. Instanciar as 3 estratégias de chunking
    print("\nSegmentando texto em chunks...")
    fixed_splitter = MeuFixedSizeChunkerTCC(chunk_size=512, chunk_overlap=51)
    recursive_splitter = MeuRecursiveChunkerTCC(chunk_size=512, chunk_overlap=51)
    semantic_splitter = MeuSemanticChunkerTCC(embeddings=embeddings, percentile_threshold=70.0)
    
    chunks_dict = {
        "fixed": fixed_splitter.split_text(text_content),
        "recursive": recursive_splitter.split_text(text_content),
        "semantic": semantic_splitter.split_text(text_content)
    }
    
    for key, val in chunks_dict.items():
        print(f"-> {key.upper()}: {len(val)} chunks gerados.")
        
    # 5. Indexar no ChromaDB
    print("\nIndexando chunks no ChromaDB (in-memory)...")
    dbs = {}
    for key, chunks in chunks_dict.items():
        docs = [Document(page_content=c, metadata={"strategy": key}) for c in chunks]
        dbs[key] = Chroma.from_documents(docs, embeddings, collection_name=f"db_{key}")
    print("Indexação concluída.")
    
    # =========================================================================
    # FASE 1: GERAÇÃO COM LLAMA 3.1 8B (SLM Gerador do TCC)
    # =========================================================================
    print("\n" + "-"*50)
    print(" FASE 1: GERAÇÃO (LLaMA 3.1 8B Instruct)")
    print("-"*50)
    
    try:
        llama_model = load_llama_model()
    except Exception as e:
        print(f"Erro ao carregar LLaMA: {e}")
        return
        
    results_by_strategy = {}
    
    for strategy, db in dbs.items():
        print(f"\nRodando geração RAG para: {strategy.upper()}")
        records = []
        
        for idx, qa in enumerate(qa_dataset):
            question = qa["question"]
            ground_truth = qa["ground_truth"]
            
            # A. Recuperação (Retrieval Top-3)
            retrieved_docs = db.similarity_search(question, k=3)
            contexts = [doc.page_content for doc in retrieved_docs]
            context_text = "\n\n".join(contexts)
            
            # B. Geração (Generation) - LLaMA 3.1
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Você é um assistente virtual agronômico de alta precisão.
Responda à pergunta do usuário baseando-se estritamente nas informações fornecidas no contexto abaixo.
Se a resposta não estiver contida no contexto, diga "Não encontrei essa informação no contexto".
Não invente nenhum fato ou valor numérico que não esteja explicitamente escrito no contexto.<|eot_id|><|start_header_id|>user<|end_header_id|>
Contexto:
{context_text}

Pergunta:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            answer = get_llama_response(prompt, llama_model)
            
            records.append({
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": ground_truth
            })
            print(f"Pergunta {idx+1}/{len(qa_dataset)} processada.")
            
        results_by_strategy[strategy] = pd.DataFrame(records)
        
    # =========================================================================
    # LIBERAR MEMÓRIA (VRAM) ANTES DE CARREGAR O QWEN
    # =========================================================================
    print("\nGeração concluída. Liberando LLaMA 3.1 da memória de vídeo (VRAM)...")
    del llama_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # =========================================================================
    # FASE 2: AVALIAÇÃO RAGAS COM QWEN 2.5 7B (Juiz/Avaliador)
    # =========================================================================
    print("\n" + "-"*50)
    print(" FASE 2: AVALIAÇÃO RAGAS (Qwen 2.5 7B Instruct)")
    print("-"*50)
    
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
        from ragas.llms import LangchainLLM
        from ragas.embeddings import LangchainEmbeddings
    except ImportError:
        print("Erro: Ragas não instalado. (!pip install ragas)")
        return
        
    try:
        qwen_model = load_qwen_model()
    except Exception as e:
        print(f"Erro ao carregar Qwen: {e}")
        return
        
    print("Configurando RAGAS para avaliação local usando Qwen 2.5 e BGE-M3...")
    evaluator_llm = LangchainLLM(llm=qwen_model)
    evaluator_embeddings = LangchainEmbeddings(embeddings=embeddings)
        
    metrics = [faithfulness, answer_relevance, context_precision, context_recall]
    ragas_scores = {}
    
    for strategy, df in results_by_strategy.items():
        print(f"\nCalculando métricas RAGAS para: {strategy.upper()}...")
        dataset = Dataset.from_pandas(df)
        
        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        
        scores_df = eval_result.to_pandas()
        scores_df["strategy"] = strategy
        ragas_scores[strategy] = scores_df
        
        print(f"\nResultados Médios - {strategy.upper()}:")
        for metric in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]:
            if metric in eval_result:
                print(f"  {metric.capitalize()}: {eval_result[metric]:.4f}")
                
    # =========================================================================
    # SALVAR RESULTADOS
    # =========================================================================
    all_scores = pd.concat(ragas_scores.values(), ignore_index=True)
    output_dir = script_dir.parent / "analises"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "ragas_evaluation_results.csv"
    all_scores.to_csv(csv_path, index=False, encoding="utf-8")
    
    summary = all_scores.groupby("strategy")[["faithfulness", "answer_relevance", "context_precision", "context_recall"]].mean()
    summary_path = output_dir / "ragas_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8")
    
    print("\n" + "="*50)
    print(" TABELA COMPARATIVA GERAL (MÉDIAS) ".center(50, "="))
    print("="*50)
    print(summary)

if __name__ == "__main__":
    run_rag_evaluation()
