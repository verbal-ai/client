# Copy service file to systemd
sudo cp src/firmware/wifi_manager.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable wifi_manager
sudo systemctl start wifi_manager