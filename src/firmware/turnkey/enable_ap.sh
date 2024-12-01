#!/bin/bash

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

function enable_ap() {
    # Stop services initially
    systemctl stop dnsmasq
    systemctl stop hostapd

    # Configure static IP
    cat > /etc/dhcpcd.conf << EOF
interface wlan0
    static ip_address=192.168.4.1/24
    nohook wpa_supplicant
EOF

    # Restart dhcpcd
    service dhcpcd restart

    # Kill any existing wpa_supplicant process
    killall wpa_supplicant 2>/dev/null || true
    
    # Unblock wifi if it's blocked
    rfkill unblock wifi

    # Enable IP forwarding
    sysctl -w net.ipv4.ip_forward=1

    # Configure IP masquerading
    iptables -t nat -F
    iptables -F
    iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
    iptables -A FORWARD -i eth0 -o wlan0 -m state --state RELATED,ESTABLISHED -j ACCEPT
    iptables -A FORWARD -i wlan0 -o eth0 -j ACCEPT
    netfilter-persistent save

    # Enable and start services
    systemctl unmask hostapd
    systemctl enable hostapd
    
    # Try starting hostapd with debug output
    echo "Starting hostapd..."
    if ! systemctl start hostapd; then
        echo "Failed to start hostapd. Checking logs..."
        journalctl -xeu hostapd.service
        hostapd -dd /etc/hostapd/hostapd.conf
    fi
    
    systemctl start dnsmasq

    echo "Access Point enabled"
    return 0
}

function disable_ap() {
    # Stop services
    systemctl stop hostapd
    systemctl stop dnsmasq

    # Remove static IP configuration
    sed -i '/interface wlan0/d' /etc/dhcpcd.conf
    sed -i '/static ip_address=192.168.4.1\/24/d' /etc/dhcpcd.conf
    sed -i '/nohook wpa_supplicant/d' /etc/dhcpcd.conf

    # Restart dhcpcd
    service dhcpcd restart

    # Clear iptables rules
    iptables -t nat -F
    iptables -F
    netfilter-persistent save

    echo "Access Point disabled"
    return 0
}

# Configure hostapd
cat > /etc/hostapd/hostapd.conf << EOF
country_code=US
interface=wlan0
driver=nl80211
ssid=RaspberryAP
hw_mode=g
channel=7
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=raspberry
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
EOF

# Make sure the configuration is readable
chmod 600 /etc/hostapd/hostapd.conf

# Check command line argument
case "$1" in
    "enable")
        enable_ap
        exit $?
        ;;
    "disable")
        disable_ap
        exit $?
        ;;
    *)
        echo "Usage: $0 {enable|disable}"
        exit 1
        ;;
esac