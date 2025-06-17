# Gunakan image python yang ringan
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Salin semua file ke dalam container
COPY . /app

# Install dependensi
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port Streamlit
EXPOSE 8501

# Jalankan aplikasi
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
