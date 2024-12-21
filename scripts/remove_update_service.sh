# Stop the service
sudo systemctl stop update

# Disable the service from starting at boot
sudo systemctl disable update

# Remove the service file
sudo rm /etc/systemd/system/update.service

# Reload systemd to recognize the changes
sudo systemctl daemon-reload

# Reset any failed status
sudo systemctl reset-failed 