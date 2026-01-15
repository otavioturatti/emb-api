"""
Módulo para geração de embeddings semânticos usando sentence-transformers.

Usa o modelo paraphrase-multilingual-MiniLM-L12-v2 que:
- Funciona bem com português
- Gera embeddings de 384 dimensões
- Entende semântica (abreviações, sinônimos, etc.)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Configurações do modelo
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# Cache global do modelo (carrega uma vez)
_model = None


def get_model():
    """
    Retorna o modelo sentence-transformer (singleton).
    Carrega na primeira chamada e reutiliza depois.
    """
    global _model
    if _model is None:
        logger.info("Carregando modelo sentence-transformers: %s", DEFAULT_MODEL)
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(DEFAULT_MODEL)
            logger.info("Modelo carregado com sucesso (dim=%d)", EMBEDDING_DIM)
        except ImportError:
            raise ImportError(
                "sentence-transformers não instalado. "
                "Execute: pip install sentence-transformers"
            )
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Gera embedding para um único texto.

    Args:
        text: Texto para gerar embedding

    Returns:
        Array numpy de 384 dimensões (float32)
    """
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return embedding.astype("float32")
