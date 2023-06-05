# Dockerfile
FROM python:3.9-slim-buster
WORKDIR /root
COPY requirements.txt /root/
RUN pip install -r requirements.txt
# Copy the entire contents of the current directory to the container
COPY . .
ENTRYPOINT ["python"]
CMD ["src/app.py"]