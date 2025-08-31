# Use official PyTorch image (already has torch installed)
FROM pytorch/pytorch:2.8.0-cpu

# Set working directory
WORKDIR /app

# Copy requirements and install extras
COPY requirements.txt .

# Install only extra dependencies, torch is already included in base
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy app code
COPY . .

# Expose port for Flask
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
