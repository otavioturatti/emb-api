"""
API FastAPI para geração de embeddings de texto.
Mantém o modelo sentence-transformer carregado em memória para respostas rápidas.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging

from sentence_embedder import embed_text, EMBEDDING_DIM

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa a API
app = FastAPI(
    title="Sentence Embedding API",
    description="API para gerar embeddings semânticos de textos usando sentence-transformers",
    version="1.0.0"
)


# Modelos de request/response
class EmbedRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "coca"
            }
        }


class EmbedResponse(BaseModel):
    embedding: List[float]
    dimensions: int

    class Config:
        json_schema_extra = {
            "example": {
                "embedding": [0.123, -0.456, 0.789],
                "dimensions": 384
            }
        }


@app.on_event("startup")
async def startup_event():
    """Carrega o modelo na inicialização da API"""
    logger.info("Iniciando API e carregando modelo...")
    from sentence_embedder import get_model
    get_model()  # Força carregamento do modelo
    logger.info("Modelo carregado! API pronta para receber requisições.")


@app.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "Sentence Embedding API",
        "status": "running",
        "embedding_dimensions": EMBEDDING_DIM,
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.post("/embed", response_model=EmbedResponse)
async def create_embedding(request: EmbedRequest):
    """
    Gera embedding para um texto.

    - **text**: Texto para gerar embedding (qualquer tamanho)
    - Retorna vetor de 384 dimensões
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Texto não pode ser vazio")

        # Gera o embedding
        embedding = embed_text(request.text)

        return EmbedResponse(
            embedding=embedding.tolist(),
            dimensions=len(embedding)
        )

    except Exception as e:
        logger.error(f"Erro ao gerar embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar embedding: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
