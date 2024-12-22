# Copy service file to systemd
BASE=/home/dev/client
sudo cp ${BASE}/services/turnkey/turnkey.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable turnkey
sudo systemctl start turnkey
