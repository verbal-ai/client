#!/bin/bash

INTERFACE="wlan0"

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
    fi
    
    sleep 5  # Wait for 5 seconds before next check
done