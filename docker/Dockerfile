# Use Python 3.12-slim as base image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files to the container
COPY . .  

# Ensure the docker directory exists inside the container
RUN mkdir -p docker

# Copy all CSV files from the "data/" folder to "docker/" inside the container
#COPY ../data/*.csv docker/

# Install dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Install required Python packages
RUN pip install --no-cache-dir \
    jupyter \
    streamlit 

# Expose the necessary ports
EXPOSE 8888 8501

# Start both services
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]