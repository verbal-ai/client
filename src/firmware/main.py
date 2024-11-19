import json
import time
import subprocess
import os

# Run the command with sudo
# sudo apt-get install dhcpcd5
# sudo systemctl enable dhcpcd
# sudo systemctl start dhcpcd


def load_wifi_config():
    try:
        # Get the directory where the script is located
        config_path = os.path.join('/home/dev/', 'wifi.json')
        
        print(f"Attempting to load config from: {config_path}")  # Debug line
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Validate the config structure
            if not config.get('ap_config') or not config.get('known_networks'):
                print("Invalid config structure in wifi.json")
                return None
            return config
    except FileNotFoundError:
        print(f"wifi.json not found at {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in wifi.json: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading wifi.json: {e}")
        return None


def enable_services():
    try:
        # Enable wpa_supplicant first
        subprocess.run(['sudo', 'systemctl', 'enable', 'wpa_supplicant'], check=True)
        
        # Check if dhcpcd service exists before enabling
        result = subprocess.run(['systemctl', 'list-unit-files', 'dhcpcd.service'], 
                              capture_output=True, 
                              text=True)
        
        if 'dhcpcd.service' in result.stdout:
            subprocess.run(['sudo', 'systemctl', 'enable', 'dhcpcd'], check=True)
        else:
            print("dhcpcd service not found, installing...")
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'dhcpcd5'], check=True)
            subprocess.run(['sudo', 'systemctl', 'enable', 'dhcpcd'], check=True)
            
    except subprocess.CalledProcessError as e:
        print(f"Error enabling services: {e}")
        return False
    return True

def check_required_packages():
    enable_services()
    required_packages = ['wireless-tools', 'wpasupplicant']
    missing_packages = []
    
    try:
        # First check which packages are missing
        for package in required_packages:
            result = subprocess.run(['dpkg', '-s', package], 
                                  capture_output=True, 
                                  text=True)
            if result.returncode != 0:
                missing_packages.append(package)
        
        # Only update and install if there are missing packages
        if missing_packages:
            print(f"Missing packages: {missing_packages}")
            print("Updating package list...")
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            
            for package in missing_packages:
                print(f"Installing {package}...")
                subprocess.run(['sudo', 'apt-get', 'install', '-y', package], check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install required packages: {e}")
        return False

def check_wifi_interface():
    try:
        # Check if wlan0 exists
        result = subprocess.run(['ip', 'link', 'show', 'wlan0'], capture_output=True, text=True)
        if result.returncode != 0:
            print("WiFi interface wlan0 not found!")
            return False
            
        # Ensure WiFi is not blocked
        subprocess.run(['sudo', 'rfkill', 'unblock', 'wifi'], check=False)
        
        # Bring interface up
        subprocess.run(['sudo', 'ip', 'link', 'set', 'wlan0', 'up'], check=True)
        time.sleep(2)  # Give interface time to come up
        
        return True
    except Exception as e:
        print(f"Error checking WiFi interface: {e}")
        return False

def cleanup_wifi():
    """Thoroughly clean up all wireless interfaces and processes"""
    try:
        # Stop all related services
        services = ['wpa_supplicant', 'dhcpcd', 'NetworkManager', 'dhclient']
        for service in services:
            subprocess.run(['sudo', 'systemctl', 'stop', service], check=False)
            subprocess.run(['sudo', 'killall', service], check=False)
        
        # Remove the problematic control interface file
        subprocess.run(['sudo', 'rm', '-rf', '/var/run/wpa_supplicant'], check=False)
        subprocess.run(['sudo', 'mkdir', '-p', '/var/run/wpa_supplicant'], check=False)
        
        # Reset the wireless interface
        subprocess.run(['sudo', 'ip', 'link', 'set', 'wlan0', 'down'], check=False)
        time.sleep(1)
        subprocess.run(['sudo', 'ip', 'link', 'set', 'wlan0', 'up'], check=False)
        time.sleep(2)
        
        return True
    except Exception as e:
        print(f"Cleanup failed: {e}")
        return False

def connect_wifi():
    if not check_required_packages():
        return False
        
    if not check_wifi_interface():
        return False
    
    wifi_config = load_wifi_config()
    if not wifi_config:
        return False
    
    # Clean up before starting
    if not cleanup_wifi():
        print("Failed to cleanup wireless interfaces")
        return False
    
    try:
        for net in wifi_config['known_networks']:
            print(f"\nTrying to connect to {net['ssid']}")
            
            # Create network configuration
            network_config = (
                'ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev\n'
                'update_config=1\n'
                'country=US\n\n'
                f'network={{\n'
                f'    ssid="{net["ssid"]}"\n'
                f'    psk="{net["password"]}"\n'
                f'    key_mgmt=WPA-PSK\n'
                f'}}\n'
            )
            
            # Write configuration
            with open('/etc/wpa_supplicant/wpa_supplicant.conf', 'w') as f:
                f.write(network_config)
            
            # Set proper permissions
            subprocess.run(['sudo', 'chmod', '600', '/etc/wpa_supplicant/wpa_supplicant.conf'])
            
            # Start wpa_supplicant with specific socket path
            subprocess.run([
                'sudo',
                'wpa_supplicant',
                '-B',                    # Run in background
                '-i', 'wlan0',          # Interface name
                '-c', '/etc/wpa_supplicant/wpa_supplicant.conf',  # Config file
                '-P', '/var/run/wpa_supplicant.pid',  # PID file
                '-D', 'nl80211,wext'    # Driver options
            ], check=False)
            
            # Wait for connection
            max_wait = 30
            while max_wait > 0:
                result = subprocess.run(['iwconfig', 'wlan0'], capture_output=True, text=True)
                if net['ssid'] in result.stdout and 'ESSID:off/any' not in result.stdout:
                    print(f"Successfully connected to {net['ssid']}")
                    
                    # Get IP address via DHCP
                    subprocess.run(['sudo', 'dhclient', 'wlan0'], check=False)
                    return True
                max_wait -= 1
                time.sleep(1)
                print("Waiting for connection...")
                
    except Exception as e:
        print(f"Connection failed: {e}")
        
    return False

def main():
    if not connect_wifi():
        print("Could not connect to any known networks")

if __name__ == '__main__':
    main()
