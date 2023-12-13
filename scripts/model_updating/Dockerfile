# Use python base image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the local requirements.txt to the container at /app 
COPY ../requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy root directory
COPY .. .

# Install PyTradeX library
RUN pip install --no-cache-dir -e .

