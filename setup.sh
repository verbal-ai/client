#!/bin/bash

# Check if python3-venv is installed
if ! dpkg -l | grep -q python3-venv; then
    sudo apt update
    sudo apt install -y python3-venv python3-full
fi

# Remove existing venv if it exists
if [ -d "venv" ]; then
    rm -rf venv
fi

# Create new virtual environment
python3 -m venv venv --without-pip

# Activate virtual environment
source venv/bin/activate

# Install pip
curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
# rm get-pip.py

pip install pdm

# Initialize PDM project if pyproject.toml doesn't exist
if [ ! -f "pyproject.toml" ]; then
    pdm init --python $(which python)
fi


# Install required packages
pip install requests flask

# Verify installations
python -c "import requests; print(f'Requests version: {requests.__version__}')"
python -c "import flask; print(f'Flask version: {flask.__version__}')"

echo "Setup complete. Use 'source venv/bin/activate' to activate the environment"



echo "Setting up systemd service"

# sudo cp src/firmware/wifi_manager.service /etc/systemd/system/
# sudo chmod 644 /etc/systemd/system/wifi_manager.service
# sudo systemctl daemon-reload
# sudo systemctl enable wifi_manager.service
# sudo systemctl start wifi_manager.service

# echo "Setup complete. Service installed and started."
# echo "Check status with: sudo systemctl status wifi_manager.service"
