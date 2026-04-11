ARG ARCH=arm64v8
ARG DISTRO=daffy
FROM duckietown/dt-core:${DISTRO}-${ARCH}

# Install ML dependencies
COPY requirements_duckiebot.txt .
RUN pip3 install --no-cache-dir -r requirements_duckiebot.txt

# Set workspace
WORKDIR /solution

# Copy your code and models
COPY . .

# Launch using the standard launcher
CMD ["python3", "solution.py"]