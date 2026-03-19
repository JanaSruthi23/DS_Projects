# Step 1: Use a lightweight Python base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy your requirements first (better for caching)
COPY requirements.txt .

# Step 4: Install your Data Science libraries
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your app code and model files
COPY . .

# Step 6: Expose the default Streamlit port
EXPOSE 8501

# Step 7: Run the app when the container starts
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
