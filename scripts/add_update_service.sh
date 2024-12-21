# Copy service file to systemd
sudo cp services/update/update.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable update
sudo systemctl start update 