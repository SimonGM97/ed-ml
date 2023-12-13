# Use python base image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the local requirements.txt to the container at /app 
COPY ../requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy required directory & files
COPY ../config/params.py .
COPY ../ed_ml .
COPY ../scripts .
COPY ../README.md .
COPY ../setup.py .

# Install ed_ml library
RUN pip install --no-cache-dir -e .

EXPOSE 5050

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5050"]

# Run the model_tuning.py script when the container launches
# CMD ["./scripts/model_tuning/model_tuning.py"]
ENTRYPOINT ["python", "./scripts/model_tuning/model_tuning.py"]
