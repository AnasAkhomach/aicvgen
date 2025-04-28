# Use a Python 3.11 base image - choose a slim version for smaller size
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the working directory
# This copies everything from your project root where the Dockerfile is located
COPY . .

# Command to run the application when the container starts
# Assuming main.py is the entry point
CMD ["python", "main.py"]

# If you want to run the Jupyter notebook, you might change the CMD or add an entrypoint
# For example, to run the Jupyter notebook server:
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
# And expose the port:
# EXPOSE 8888
