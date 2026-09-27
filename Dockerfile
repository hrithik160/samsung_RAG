FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy lockfile and install dependencies first to leverage Docker cache
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# Copy the rest of the project files
COPY . .

# Default command to run the evaluation
CMD ["python", "component5/run_eval.py"]
