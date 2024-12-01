# Stop the service
sudo systemctl stop turnkey

# Disable the service from starting at boot
sudo systemctl disable turnkey

# Remove the service file
sudo rm /etc/systemd/system/turnkey.service

# Reload systemd to recognize the changes
sudo systemctl daemon-reload

# Reset any failed status
sudo systemctl reset-failed