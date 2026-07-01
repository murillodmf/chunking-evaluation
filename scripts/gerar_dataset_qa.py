import os
import json
import re
from pathlib import Path
import google.generativeai as genai
from pypdf import PdfReader

def load_and_clean_pdf(pdf_path: str) -> str:
    """
    Carrega o PDF de ecofisiologia da soja e aplica limpeza de ruídos.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF não encontrado em: {path.absolute()}")
    
    print(f"Lendo PDF de: {path.absolute()}")
    reader = PdfReader(str(path))
    
    extracted_text = []
    # Começa na página 6 (índice 5, onde inicia a seção relevante)
    for page_num in range(5, len(reader.pages)):
        page_text = reader.pages[page_num].extract_text()
        if page_text:
            extracted_text.append(page_text)
            
    full_text = "\n".join(extracted_text)
    
    # 1. Remover legendas de figuras
    fig_pattern = r'Figura \d+\..*?\.(?=\s*\n|\s*$)'
    full_text = re.sub(fig_pattern, '', full_text, flags=re.DOTALL)
    
    # 2. Filtrar linhas de cabeçalho, rodapé, créditos de imagem e numerações
    cleaned_lines = []
    for line in full_text.split('\n'):
        l = line.strip()
        if not l:
            continue
        if re.match(r'^\d+$', l):
            continue
        if "sistemas de produ" in l.lower() or "tecnologias de produ" in l.lower():
            continue
        if l.lower().startswith("foto:") or l.lower().startswith("fotos:") or "bouças farias" in l.lower() or "balbinot junior" in l.lower() or "arrabal arias" in l.lower():
            continue
        if re.match(r'^[a-e](\s+[a-e])*$', l.lower()):
            continue
        cleaned_lines.append(line)
        
    full_text = "\n".join(cleaned_lines)
    full_text = re.sub(r'(\w+)\s*-\s*\n\s*(\w+)', r'\1\2', full_text)
    
    # Iniciar na seção "Exigências climáticas"
    target_start = "Exigências climáticas"
    start_idx = full_text.find(target_start)
    if start_idx != -1:
        full_text = full_text[start_idx:]
        
    # Remover referências bibliográficas do final
    ref_target = "Referências"
    ref_idx = full_text.find(ref_target)
    if ref_idx != -1:
        full_text = full_text[:ref_idx]
        
    return full_text

def split_text_into_large_blocks(text: str, block_size_chars: int = 5000) -> list:
    """
    Divide o texto em blocos grandes de aproximadamente 5000 caracteres para 
    geração de perguntas contextualizadas.
    """
    blocks = []
    current_idx = 0
    while current_idx < len(text):
        end_idx = min(current_idx + block_size_chars, len(text))
        # Ajustar para o final do parágrafo mais próximo para não cortar frases no meio
        if end_idx < len(text):
            next_newline = text.find("\n\n", end_idx)
            if next_newline != -1 and next_newline < end_idx + 1000:
                end_idx = next_newline + 2
        blocks.append(text[current_idx:end_idx].strip())
        current_idx = end_idx
    return blocks

def generate_qa_pairs_for_block(block: str, gemini_client) -> list:
    """
    Usa o Gemini para gerar perguntas e respostas técnicas baseadas no bloco de texto.
    """
    prompt = f"""
Você é um especialista em Agronomia e Inteligência Artificial.
Baseando-se exclusivamente no texto técnico da Embrapa fornecido abaixo, crie de 3 a 5 perguntas técnicas de alta qualidade acompanhadas de suas respectivas respostas de referência (ground_truth).

Regras obrigatórias:
1. As perguntas devem ser extremamente específicas e detalhadas, típicas de provas ou consultas técnicas de agrônomos (ex: evitar perguntas genéricas como "O que é soja?", prefira "Qual o efeito de temperaturas do solo abaixo de 20°C na germinação da soja?").
2. A resposta de referência (ground_truth) deve ser detalhada, completa e conter todos os fatos e dados numéricos presentes no texto que respondam à pergunta.
3. Retorne a resposta estritamente formatada como um array JSON válido de objetos, sem blocos de código adicionais (como ```json) ou introduções. Cada objeto do array deve ter as chaves "question" e "ground_truth".

Exemplo de formato esperado:
[
  {{
    "question": "Pergunta 1...",
    "ground_truth": "Resposta 1..."
  }},
  {{
    "question": "Pergunta 2...",
    "ground_truth": "Resposta 2..."
  }}
]

Texto de referência:
---
{block}
---
"""
    try:
        response = gemini_client.generate_content(prompt)
        text_response = response.text.strip()
        
        # Limpar markdown do JSON se o modelo teimar em retornar com ```json
        if text_response.startswith("```json"):
            text_response = text_response[7:]
        if text_response.endswith("```"):
            text_response = text_response[:-3]
        text_response = text_response.strip()
        
        qa_pairs = json.loads(text_response)
        return qa_pairs
    except Exception as e:
        print(f"Erro ao gerar perguntas para o bloco: {e}")
        return []

def main():
    # 1. Configurar API do Gemini
    # No Colab, o usuário deve setar a API key nas "Secrets" com o nome GEMINI_API_KEY
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("AVISO: GEMINI_API_KEY não encontrada nas variáveis de ambiente.")
        print("Insira a chave do Gemini manualmente abaixo para prosseguir (ou configure no Colab Secrets):")
        api_key = input("API Key do Gemini: ").strip()
        
    if not api_key:
        print("Erro: Chave de API necessária.")
        return
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # 2. Carregar texto do PDF
    script_dir = Path(__file__).parent.resolve()
    pdf_path = script_dir / "ecofisiologia_soja_embrapa.pdf"
    
    # Caso esteja rodando no repositório clonado no Colab
    if not pdf_path.exists():
        pdf_path = script_dir.parent / "Textos_exemplo" / "ecofisiologia_soja_embrapa.pdf"
        
    if not pdf_path.exists():
        pdf_path = Path("Textos_exemplo/ecofisiologia_soja_embrapa.pdf")
        
    try:
        text_content = load_and_clean_pdf(str(pdf_path))
        print(f"Texto do PDF carregado. Tamanho: {len(text_content)} caracteres.")
    except Exception as e:
        print(f"Erro ao ler PDF: {e}")
        return
        
    # 3. Dividir texto em blocos grandes
    blocks = split_text_into_large_blocks(text_content)
    print(f"Texto dividido em {len(blocks)} blocos de contexto.")
    
    # 4. Iterar e gerar as perguntas
    all_qa_pairs = []
    for idx, block in enumerate(blocks):
        print(f"Gerando perguntas para o bloco {idx+1}/{len(blocks)}...")
        block_qa = generate_qa_pairs_for_block(block, model)
        print(f"-> Gerou {len(block_qa)} pares de perguntas e respostas.")
        all_qa_pairs.extend(block_qa)
        
    print(f"\nGeração concluída! Total de {len(all_qa_pairs)} pares de QA gerados.")
    
    # 5. Salvar resultado em JSON
    output_dir = script_dir.parent / "Textos_exemplo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qa_dataset.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)
        
    print(f"Dataset de QA salvo com sucesso em: {output_path.absolute()}")

if __name__ == "__main__":
    main()
