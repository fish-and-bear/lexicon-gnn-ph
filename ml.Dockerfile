# Use a stable, official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
# This is done as a separate step to leverage Docker's layer caching
COPY ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ml/ ./ml
COPY ml/db_config.json ./ml/db_config.json

# Command to run the fine-tuning script
CMD ["python", "ml/final_production_system.py"] 