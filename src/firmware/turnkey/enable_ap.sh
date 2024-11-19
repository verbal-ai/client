#!/bin/bash

# Function to check if a package is installed
check_package() {
    if ! dpkg -l | grep -q "^ii  $1 "; then
        echo "Installing $1..."
        sudo apt-get update
        sudo apt-get install -y $1
        return 0
    else
        echo "$1 is already installed"
        return 1
    fi
}

echo "Preparing WiFi interface..."
# Unblock WiFi if blocked
sudo rfkill unblock wifi
sudo rfkill unblock wlan

# Reset wlan0 interface
sudo ip link set wlan0 down || true
sudo ip addr flush dev wlan0 || true
sleep 2

# Check and install required packages
check_package hostapd
HOSTAPD_INSTALLED=$?

check_package dnsmasq
DNSMASQ_INSTALLED=$?

# If we installed any packages, enable the services
if [ $HOSTAPD_INSTALLED -eq 0 ] || [ $DNSMASQ_INSTALLED -eq 0 ]; then
    echo "Enabling services..."
    sudo systemctl unmask hostapd
    sudo systemctl enable hostapd
    sudo systemctl enable dnsmasq
fi

# Create required directories
sudo mkdir -p /etc/hostapd
sudo mkdir -p /etc/default/hostapd

# Stop services
echo "Stopping services..."
# sudo systemctl stop wpa_supplicant
# sudo systemctl stop dhcpcd
sudo systemctl stop dnsmasq || true
sudo systemctl stop hostapd || true

# Copy configurations
echo "Copying configurations..."
sudo cp ./config/hostapd /etc/default/hostapd
sudo cp ./config/hostapd.conf /etc/hostapd/hostapd.conf
# sudo cp ./config/dhcpcd.conf /etc/dhcpcd.conf
sudo cp ./config/dnsmasq.conf /etc/dnsmasq.conf

# Set permissions
sudo chmod 600 /etc/hostapd/hostapd.conf

# Load wan configuration
# sudo cp wpa.conf /etc/wpa_supplicant/wpa_supplicant.conf

# Start services
echo "Starting services..."
# sudo systemctl start dhcpcd
sudo systemctl start dnsmasq
sudo systemctl start hostapd

echo "Configuration complete. Rebooting..."
# sudo reboot now

