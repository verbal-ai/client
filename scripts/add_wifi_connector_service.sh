# Copy service file to systemd
sudo cp services/wifi_connector/wifi_connector.service /etc/systemd/system/

sudo systemctl unmask wifi_connector
# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl start wifi_connector
sudo systemctl enable wifi_connector
