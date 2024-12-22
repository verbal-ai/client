# Copy service file to systemd
BASE=/home/dev/client
sudo cp ${BASE}/services/update/update.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable update
sudo systemctl start update 