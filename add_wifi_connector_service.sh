# Copy service file to systemd
sudo cp services/wifi_connector/wifi_connector.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable wifi_connector
sudo systemctl start wifi_connector
