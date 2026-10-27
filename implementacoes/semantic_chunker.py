import re
import time
import numpy as np
from typing import List, Any
from langchain_text_splitters import TextSplitter

_SENT_DIST_CACHE = {}

class MeuSemanticChunkerTCC(TextSplitter):
    """
    Subclasse customizada para o TCC que implementa a divisão semântica baseada em embeddings
    e distância de cosseno entre sentenças vizinhas.
    """
    def __init__(self, embeddings: Any, percentile_threshold: float = 70.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.embeddings = embeddings
        self.percentile_threshold = percentile_threshold
        self.last_standalone_seconds = 0.0

    def split_text(self, text: str) -> List[str]:
        """
        Divide o texto com base no limite estatístico das distâncias semânticas das sentenças.
        """
        if not text or not text.strip():
            self.last_standalone_seconds = 0.0
            return []

        if text in _SENT_DIST_CACHE:
            sentences, distances, base_elapsed = _SENT_DIST_CACHE[text]
        else:
            t0 = time.perf_counter()
            # 1. Dividir em sentenças usando expressões regulares
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            
            if not sentences:
                self.last_standalone_seconds = time.perf_counter() - t0
                return []
            if len(sentences) == 1:
                self.last_standalone_seconds = time.perf_counter() - t0
                return sentences

            # 2. Gerar embeddings para cada sentença
            embeddings_list = self.embeddings.embed_documents(sentences)
            
            # 3. Calcular a distância de cosseno (1 - cosine_similarity) entre vizinhos
            distances = []
            for i in range(len(embeddings_list) - 1):
                emb1 = np.array(embeddings_list[i])
                emb2 = np.array(embeddings_list[i+1])
                
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                if norm1 == 0 or norm2 == 0:
                    similarity = 0.0
                else:
                    similarity = np.dot(emb1, emb2) / (norm1 * norm2)
                
                similarity = np.clip(similarity, -1.0, 1.0)
                distances.append(1.0 - similarity)
            base_elapsed = time.perf_counter() - t0
            _SENT_DIST_CACHE[text] = (sentences, distances, base_elapsed)

        if not sentences:
            self.last_standalone_seconds = 0.0
            return []
        if len(sentences) == 1:
            self.last_standalone_seconds = base_elapsed
            return sentences

        t0_cut = time.perf_counter()
        # 4. Definir ponto de corte com base em percentil estatístico
        threshold = np.percentile(distances, self.percentile_threshold)

        # 5. Agrupar sentenças em blocos baseando-se no breakpoint
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(len(distances)):
            if distances[i] > threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i+1]]
            else:
                current_chunk.append(sentences[i+1])
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        self.last_standalone_seconds = base_elapsed + (time.perf_counter() - t0_cut)
        return chunks
