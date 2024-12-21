#!/bin/bash

# Set up logging
exec 1> >(logger -s -t $(basename $0)) 2>&1

echo "Starting update process..."

# Directory containing the service scripts
SCRIPTS_DIR="$HOME/client/scripts"

echo "Removing existing services..."
sudo bash "$SCRIPTS_DIR/remove_turnkey_service.sh"
sudo bash "$SCRIPTS_DIR/remove_voice_assistant_service.sh"
sudo bash "$SCRIPTS_DIR/remove_wifi_connector_service.sh"

echo "Adding services..."
sudo bash "$SCRIPTS_DIR/add_turnkey_service.sh"
sudo bash "$SCRIPTS_DIR/add_voice_assistant_service.sh"
sudo bash "$SCRIPTS_DIR/add_wifi_connector_service.sh"

echo "Service update completed"

# Reload systemd to apply changes
sudo systemctl daemon-reload

exit 0
