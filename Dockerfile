FROM python:3.10-slim

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5002
#CMD ["python", "app.py"]

CMD ["gunicorn", "--bind", "0.0.0.0:5002", "app:app"]