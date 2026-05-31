import tiktoken
from typing import Any
from langchain_text_splitters import CharacterTextSplitter

class MeuFixedSizeChunkerTCC(CharacterTextSplitter):
    """
    Subclasse customizada para o TCC que encapsula o splitter de tamanho fixo,
    medindo o tamanho do chunk em tokens (via tiktoken).
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 51, encoding_name: str = "cl100k_base", **kwargs: Any):
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        
        super().__init__(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length,
            **kwargs
        )

    def _token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

