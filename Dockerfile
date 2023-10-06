FROM python:3.6
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
# Install TensorFlow (or any other additional packages)
RUN pip install tensorflow --no-cache-dir

# CMD ["python", "app.py"]