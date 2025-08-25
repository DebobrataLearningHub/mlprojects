FROM python:3.12-slim
EXPOSE 8088
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . ./
ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8088", "--server.address=0.0.0.0" ]