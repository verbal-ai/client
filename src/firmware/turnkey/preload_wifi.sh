#!/bin/bash


sudo rfkill unblock wifi
sudo rfkill unblock all

sudo ip link set wlan0 up
# Ensure source file exists
if [ ! -f wpa.conf ]; then
    echo "Error: wpa.conf not found"
    exit 1
fi

# Create directory if it doesn't exist
sudo mkdir -p /etc/wpa_supplicant

# Copy with verbose output
sudo cp -v wpa.conf /etc/wpa_supplicant/wpa_supplicant.conf

# Set proper permissions
sudo chmod 600 /etc/wpa_supplicant/wpa_supplicant.conf

# Restart services
sudo systemctl restart wpa_supplicant
sudo systemctl restart dhcpcd

# Verify file exists
if [ -f /etc/wpa_supplicant/wpa_supplicant.conf ]; then
    echo "wpa_supplicant.conf created successfully"
else
    echo "Error: wpa_supplicant.conf not created"
    exit 1
fi