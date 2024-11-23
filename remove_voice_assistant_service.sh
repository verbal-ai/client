# Stop the service
sudo systemctl stop voice_assistant

# Disable the service from starting at boot
sudo systemctl disable voice_assistant

# Remove the service file
sudo rm /etc/systemd/system/voice_assistant.service

# Reload systemd to recognize the changes
sudo systemctl daemon-reload

# Reset any failed status
sudo systemctl reset-failed
