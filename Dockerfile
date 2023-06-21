FROM python:3.6
WORKDIR /app
COPY requirements.txt /app  # Copy requirements.txt first
RUN pip install -r requirements.txt  # Install dependencies
COPY . /app  # Copy the rest of the files
# CMD ["python", "app.py"]
