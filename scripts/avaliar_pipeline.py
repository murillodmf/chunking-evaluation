import os
import json
import pandas as pd
from pathlib import Path

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


def load_qwen_model():
    """
    Carrega o modelo Qwen 2.5 7B Instruct localmente com quantização de 4 bits.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
    from langchain_huggingface import HuggingFacePipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError(
            "CUDA (GPU) não está disponível. O carregamento do Qwen local em 4 bits requer GPU.\n"
            "Verifique se o runtime está configurado com aceleração de GPU (ex: T4 no Colab)."
        )
        
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Carregando tokenizer para {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Configurando quantização de 4 bits (bitsandbytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print(f"Carregando modelo {model_id} (isso pode levar alguns minutos na primeira vez)...")
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

def get_qwen_response(prompt: str, qwen_pipeline) -> str:
    """
    Gera a resposta do RAG utilizando o Qwen 2.5 local.
    """
    try:
        response = qwen_pipeline.invoke(prompt)
        if prompt in response:
            response = response.replace(prompt, "")
        return response.strip()
    except Exception as e:
        print(f"Erro ao gerar resposta com o Qwen: {e}")
        return "Erro na geração da resposta."

def run_rag_evaluation():
    # 1. Inicializar o Qwen 2.5 7B local (100% local, sem APIs)
    print("="*60)
    print(" AVALIAÇÃO RAG - 100% LOCAL (Qwen 2.5 7B + BGE-M3)")
    print("="*60)
    
    try:
        qwen_model = load_qwen_model()
        print("✅ Qwen 2.5 7B carregado com sucesso!\n")
    except Exception as e:
        print(f"\n[FALHA] Erro ao inicializar o Qwen local: {e}")
        print("Verifique se o runtime está configurado com GPU (T4).")
        return
    
    # 2. Carregar o Dataset de QA
    qa_path = script_dir.parent / "Textos_exemplo" / "qa_dataset.json"
    if not qa_path.exists():
        qa_path = Path("Textos_exemplo/qa_dataset.json")
        
    if not qa_path.exists():
        print(f"Erro: Dataset de QA não encontrado em {qa_path.absolute()}. Execute 'gerar_dataset_qa.py' primeiro.")
        return
        
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
        
    print(f"Dataset de QA carregado. Total de perguntas: {len(qa_dataset)}")
    
    # Reduzir quantidade para teste rápido se o usuário desejar (ex: top 10 perguntas)
    print("Deseja rodar para todas as perguntas ou apenas uma amostra de teste (ex: 5 perguntas)?")
    amostra_opt = input("Digite 'todas' ou o número de perguntas (ex: 5): ").strip().lower()
    if amostra_opt != "todas" and amostra_opt.isdigit():
        qa_dataset = qa_dataset[:int(amostra_opt)]
        print(f"Rodando teste para {len(qa_dataset)} perguntas.")

    # 3. Carregar e limpar texto do PDF
    pdf_path = script_dir.parent / "Textos_exemplo" / "ecofisiologia_soja_embrapa.pdf"
    if not pdf_path.exists():
        pdf_path = Path("Textos_exemplo/ecofisiologia_soja_embrapa.pdf")
        
    text_content = load_and_clean_pdf(str(pdf_path))
    
    # 4. Inicializar modelo de Embedding (BGE-M3)
    import torch
    print("\nInicializando HuggingFaceEmbeddings (BAAI/bge-m3)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    # 5. Instanciar as 3 estratégias de chunking
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
        
    # 6. Indexar no ChromaDB
    print("\nIndexando chunks no ChromaDB (in-memory)...")
    dbs = {}
    for key, chunks in chunks_dict.items():
        docs = [Document(page_content=c, metadata={"strategy": key}) for c in chunks]
        dbs[key] = Chroma.from_documents(docs, embeddings, collection_name=f"db_{key}")
    print("Indexação concluída.")
    
    # 7. Executar o pipeline RAG para cada estratégia e coletar outputs
    results_by_strategy = {}
    
    for strategy, db in dbs.items():
        print(f"\nRodando RAG para a estratégia: {strategy.upper()}")
        records = []
        
        for idx, qa in enumerate(qa_dataset):
            question = qa["question"]
            ground_truth = qa["ground_truth"]
            
            # A. Recuperação (Retrieval Top-3)
            retrieved_docs = db.similarity_search(question, k=3)
            contexts = [doc.page_content for doc in retrieved_docs]
            context_text = "\n\n".join(contexts)
            
            # B. Geração (Generation) - Qwen 2.5 local
            prompt = f"""<|im_start|>system
Você é um assistente virtual agronômico de alta precisão.
Responda à pergunta do usuário baseando-se estritamente nas informações fornecidas no contexto abaixo.
Se a resposta não estiver contida no contexto, diga "Não encontrei essa informação no contexto".
Não invente nenhum fato ou valor numérico que não esteja explicitamente escrito no contexto.<|im_end|>
<|im_start|>user
Contexto:
{context_text}

Pergunta:
{question}<|im_end|>
<|im_start|>assistant
"""
            answer = get_qwen_response(prompt, qwen_model)
            
            records.append({
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": ground_truth
            })
            
            print(f"Pergunta {idx+1}/{len(qa_dataset)} processada.")
            
        results_by_strategy[strategy] = pd.DataFrame(records)
        
    # 8. Rodar Avaliação RAGAS
    print("\n" + "="*50)
    print(" INICIANDO AVALIAÇÃO RAGAS ".center(50, "="))
    print("="*50)
    
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
        # Para Ragas usar modelos locais
        from ragas.llms import LangchainLLM
        from ragas.embeddings import LangchainEmbeddings
    except ImportError:
        print("Erro: Ragas ou dependências não instaladas. No Colab, certifique-se de rodar:")
        print("!pip install ragas datasets pandas")
        return
        
    # Configurar avaliadores 100% locais
    print("Configurando RAGAS para avaliação local usando Qwen 2.5 e BGE-M3...")
    evaluator_llm = LangchainLLM(llm=qwen_model)
    evaluator_embeddings = LangchainEmbeddings(embeddings=embeddings)
        
    metrics = [faithfulness, answer_relevance, context_precision, context_recall]
    
    ragas_scores = {}
    
    for strategy, df in results_by_strategy.items():
        print(f"\nCalculando métricas RAGAS para: {strategy.upper()}...")
        dataset = Dataset.from_pandas(df)
        
        # Executar o Ragas
        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        
        scores_df = eval_result.to_pandas()
        scores_df["strategy"] = strategy
        ragas_scores[strategy] = scores_df
        
        # Mostrar médias
        print(f"\nResultados Médios - {strategy.upper()}:")
        for metric in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]:
            if metric in eval_result:
                print(f"  {metric.capitalize()}: {eval_result[metric]:.4f}")
                
    # 9. Salvar resultados e gerar relatório comparativo
    all_scores = pd.concat(ragas_scores.values(), ignore_index=True)
    output_dir = script_dir.parent / "analises"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "ragas_evaluation_results.csv"
    all_scores.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nResultados detalhados por pergunta salvos em: {csv_path.absolute()}")
    
    # Resumo Geral
    summary = all_scores.groupby("strategy")[["faithfulness", "answer_relevance", "context_precision", "context_recall"]].mean()
    summary_path = output_dir / "ragas_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8")
    
    print("\n" + "="*50)
    print(" TABELA COMPARATIVA GERAL (MÉDIAS) ".center(50, "="))
    print("="*50)
    print(summary)
    print(f"\nResumo salvo em: {summary_path.absolute()}")
    
    # Script opcional para plotar boxplots usando matplotlib/seaborn
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Configurar estilo
        sns.set_theme(style="whitegrid")
        metrics_list = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
        
        plt.figure(figsize=(14, 10))
        for i, metric in enumerate(metrics_list, 1):
            plt.subplot(2, 2, i)
            sns.boxplot(x="strategy", y=metric, data=all_scores, palette="Set2")
            plt.title(f"Distribuição de {metric.capitalize()} por Estratégia")
            plt.ylim(-0.05, 1.05)
            
        plot_path = output_dir / "ragas_comparison_boxplots.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        print(f"Gráfico de boxplots comparativos salvo em: {plot_path.absolute()}")
    except Exception as e:
        print(f"Não foi possível gerar os gráficos (certifique-se de que matplotlib/seaborn estão instalados): {e}")

if __name__ == "__main__":
    run_rag_evaluation()
