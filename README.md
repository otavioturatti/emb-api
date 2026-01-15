# Sentence Embedding API

API FastAPI para gerar embeddings semânticos de textos usando sentence-transformers.

O modelo fica **sempre carregado em memória** para respostas ultra-rápidas.

## Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como usar

### 1. Iniciar a API

```bash
python api.py
```

A API estará disponível em: `http://localhost:8000`

### 2. Testar no navegador

Acesse a documentação interativa: `http://localhost:8000/docs`

### 3. Fazer requisições

**Exemplo com curl:**
```bash
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "coca"}'
```

**Exemplo com Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/embed",
    json={"text": "coca"}
)

data = response.json()
embedding = data["embedding"]  # Lista com 384 valores
dimensions = data["dimensions"]  # 384

print(f"Embedding gerado com {dimensions} dimensões")
```

**Exemplo no n8n:**
1. Adicione um nó "HTTP Request"
2. Configure:
   - Method: POST
   - URL: `http://localhost:8000/embed`
   - Body: JSON
   - JSON: `{"text": "{{ $json.seu_campo_de_texto }}"}`
3. O retorno terá:
   - `embedding`: array com 384 valores
   - `dimensions`: 384

## Endpoints

### `GET /`
Informações da API

### `GET /health`
Health check

### `POST /embed`
Gera embedding para um texto

**Request:**
```json
{
  "text": "seu texto aqui"
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, 0.789, ...],
  "dimensions": 384
}
```

## Características

- Modelo: `paraphrase-multilingual-MiniLM-L12-v2`
- Dimensões: 384
- Suporte: Português e múltiplos idiomas
- Performance: Milissegundos por requisição (após primeira carga)
- Modelo carregado uma única vez no startup

## Notas

- Na primeira execução, o modelo será baixado (~150 MB)
- Nas próximas vezes, usa o cache local
- Mantenha a API rodando para evitar recarregar o modelo
