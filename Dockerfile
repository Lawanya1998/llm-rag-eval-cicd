# 1. Base image — ek ready-made Python wala box uthao
FROM python:3.11-slim

# 2. Container ke andar kaam karne ki jagah set karo
WORKDIR /app

# 3. Pehle sirf requirements copy karo (smart trick — neeche samjhaya)
COPY requirements.txt .

# 4. Dependencies install karo
RUN pip install --no-cache-dir -r requirements.txt

# 5. Ab baaki saara code copy karo (rag.py, eval.py, gate.py, knowledge_base/ etc.)
COPY . .

# 6. Default command — jab container chale to kya karे
CMD ["python", "eval.py"]