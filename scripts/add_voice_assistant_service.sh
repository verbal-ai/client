# Copy service file to systemd
sudo cp services/voice_assistant/voice_assistant.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable voice_assistant
sudo systemctl start voice_assistant
