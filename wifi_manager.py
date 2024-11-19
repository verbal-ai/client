from flask import Flask, render_template, request
import subprocess
import socket
# import json
# import os
# import time

app = Flask(__name__)

# WiFi configuration template
wpa_conf = """country=IN
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="%s"
    %s
}"""

def wificonnected():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        s.close()
        return True
    except:
        return False

def check_wifi_cred(ssid, password):
    # Basic validation - you might want to enhance this
    if len(password) >= 8 or len(password) == 0:  # 0 for open networks
        return True
    return False

@app.route('/')
def index():
    # Scan for available networks
    try:
        scan_result = subprocess.check_output(["sudo", "iwlist", "wlan0", "scan"])
        ssids = []
        for line in scan_result.decode('utf-8').split('\n'):
            if 'ESSID' in line:
                ssid = line.split('ESSID:"')[1].split('"')[0]
                if ssid and ssid not in ssids:
                    ssids.append(ssid)
    except:
        ssids = []
    
    return render_template('index.html', ssids=ssids)

@app.route('/configure_wifi', methods=['POST'])
def configure_wifi():
    ssid = request.form['ssid']
    password = request.form['password']

    if not check_wifi_cred(ssid, password):
        return render_template('index.html', message="Invalid credentials!")

    # Create WPA supplicant configuration
    pwd = 'psk="' + password + '"' if password else "key_mgmt=NONE"
    wpa_config = wpa_conf % (ssid, pwd)

    # Write the configuration
    with open('/etc/wpa_supplicant/wpa_supplicant.conf', 'w') as f:
        f.write(wpa_config)

    # Restart networking
    subprocess.run(["sudo", "systemctl", "restart", "wpa_supplicant"])
    subprocess.run(["sudo", "systemctl", "restart", "dhcpcd"])

    return render_template('index.html', message="WiFi configuration updated. Please wait for connection...")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)