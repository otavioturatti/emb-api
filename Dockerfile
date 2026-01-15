FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema necessárias para sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de dependências
COPY requirements.txt .

# Atualizar pip
RUN pip install --upgrade pip

# Instalar dependências em etapas para evitar timeout
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir pydantic==2.5.3
RUN pip install --no-cache-dir fastapi==0.109.0
RUN pip install --no-cache-dir uvicorn[standard]==0.27.0
RUN pip install --no-cache-dir sentence-transformers==2.3.1

# Copiar código da aplicação
COPY . .

# Expor porta
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["python", "api.py"]
