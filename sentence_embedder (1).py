"""
Módulo para geração de embeddings semânticos usando sentence-transformers.

Usa o modelo paraphrase-multilingual-MiniLM-L12-v2 que:
- Funciona bem com português
- Gera embeddings de 384 dimensões
- Entende semântica (abreviações, sinônimos, etc.)
"""

import logging
from typing import List, Union
import numpy as np

from config import constants

logger = logging.getLogger(__name__)

# Re-exporta constantes para compatibilidade com imports existentes
DEFAULT_MODEL = constants.DEFAULT_EMBEDDING_MODEL
EMBEDDING_DIM = constants.EMBEDDING_DIM

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


def embed_texts(texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """
    Gera embeddings para múltiplos textos em batch.

    Args:
        texts: Lista de textos
        batch_size: Tamanho do batch para processamento
        show_progress: Mostrar barra de progresso

    Returns:
        Matriz numpy (N, 384) com embeddings
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    return embeddings.astype("float32")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula similaridade de cosseno entre dois vetores.

    Args:
        vec1: Primeiro vetor
        vec2: Segundo vetor

    Returns:
        Similaridade (0 a 1)
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def find_most_similar(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int = 10,
    min_similarity: float = 0.5
) -> List[tuple]:
    """
    Encontra os vetores mais similares ao query.

    Args:
        query_embedding: Vetor da query (384,)
        corpus_embeddings: Matriz de vetores (N, 384)
        top_k: Número de resultados
        min_similarity: Similaridade mínima

    Returns:
        Lista de tuplas (index, score) ordenada por score decrescente
    """
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

    # Reshape query para (1, 384)
    query = query_embedding.reshape(1, -1)

    # Calcula similaridades
    similarities = sklearn_cosine(query, corpus_embeddings)[0]

    # Ordena por similaridade decrescente
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score >= min_similarity:
            results.append((int(idx), float(score)))
        else:
            break

    return results
