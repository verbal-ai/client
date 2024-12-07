#!/bin/bash

INTERFACE="wlan0"

unblock_wifi() {
    echo "Checking RF-kill status..."
    if rfkill list wifi | grep -q "blocked: yes"; then
        echo "WiFi is blocked by RF-kill, attempting to unblock..."
        rfkill unblock wifi
        sleep 2
    fi
}

stop_wpa_supplicant() {
    # Find and kill wpa_supplicant processes specifically for wlan0
    pid=$(pgrep -f "wpa_supplicant.*$INTERFACE")
    if [ ! -z "$pid" ]; then
        echo "Stopping existing wpa_supplicant for $INTERFACE (PID: $pid)"
        kill $pid
        sleep 1
    fi
}

while true; do

    unblock_wifi
    # Check if we're already connected
    if ! iw dev $INTERFACE link | grep -q "Connected"; then
        echo "Not connected. Attempting to connect..."
        
        # Stop any existing wpa_supplicant for this interface
        stop_wpa_supplicant
        
        # Bring interface up
        ip link set $INTERFACE up
        
        # Start wpa_supplicant using existing configuration
        wpa_supplicant -B -i $INTERFACE -c /etc/wpa_supplicant/wpa_supplicant.conf -P /var/run/wpa_supplicant.pid -D nl80211,wext
        
        # Get IP using DHCP
        dhclient $INTERFACE
    else
        # Print network details when connected
        echo "Connected to WiFi. Network details:"
        echo "=================================="
        echo "Connection Info:"
        iw dev $INTERFACE link
        echo "IP Configuration:"
        ip addr show $INTERFACE
        echo "Signal Strength:"
        iwconfig $INTERFACE | grep -i "signal"
        echo "Route Information:"
        ip route | grep $INTERFACE
        echo "=================================="
    fi
    
    sleep 10  # Wait for 10 seconds before next check
done