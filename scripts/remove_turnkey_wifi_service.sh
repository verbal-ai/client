# Stop the service
sudo systemctl stop wifi_manager

# Disable the service from starting at boot
sudo systemctl disable wifi_manager

# Remove the service file
sudo rm /etc/systemd/system/wifi_manager.service

# Reload systemd to recognize the changes
sudo systemctl daemon-reload

# Reset any failed status
sudo systemctl reset-failed