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
COPY ../app.py .
COPY ../README.md .
COPY ../setup.py .

# Install ed_ml library
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8501

# Run the app.py script when the container launches
CMD ["streamlit", "run", "app.py"]