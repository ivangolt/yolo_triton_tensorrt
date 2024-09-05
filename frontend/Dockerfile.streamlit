FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    requests==2.32.3 \
    streamlit==1.22.0 \
    plotly==5.14.1

EXPOSE 8501

COPY app.py /app/

CMD streamlit run --server.port 8501 --server.enableCORS false app.py
