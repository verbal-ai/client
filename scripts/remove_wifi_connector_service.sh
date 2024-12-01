# Stop the service
sudo systemctl stop wifi_connector

# Disable the service from starting at boot
sudo systemctl disable wifi_connector

# Remove the service file
sudo rm /etc/systemd/system/wifi_connector.service

# Reload systemd to recognize the changes
sudo systemctl daemon-reload

# Reset any failed status
sudo systemctl reset-failed
