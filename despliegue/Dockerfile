FROM python:3.12-slim

WORKDIR /app

COPY despliegue/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY despliegue/ .
COPY modelos/ ./modelos/

EXPOSE 8050

CMD ["python", "tablero.py"]