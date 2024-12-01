import json
import time
import subprocess
import os
import asyncio
import string
import random

# Run the command with sudo
# sudo apt-get install dhcpcd5
# sudo systemctl enable dhcpcd
# sudo systemctl start dhcpcd
from flask import Flask, render_template, redirect
app = Flask(__name__, static_url_path='')

currentdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(currentdir)

ssid_list = []
def getssid():
    global ssid_list
    if len(ssid_list) > 0:
        return ssid_list
    ssid_list = []
    
    try:
        # Initiate scan
        subprocess.run(['sudo', 'wpa_cli', 'scan'], check=True)
        time.sleep(2)  # Wait for scan to complete
        
        # Get scan results
        scan_results = subprocess.check_output(['sudo', 'wpa_cli', 'scan_results'], 
                                             text=True)
        
        # Parse results
        for line in scan_results.splitlines()[1:]:  # Skip header line
            try:
                # Format: bssid / frequency / signal level / flags / ssid
                parts = line.split('\t')
                if len(parts) >= 5:
                    ssid = parts[4].strip()
                    if ssid and ssid not in ssid_list:  # Only add non-empty SSIDs
                        ssid_list.append(ssid)
            except:
                continue
                
    except subprocess.CalledProcessError as e:
        print(f"Error scanning networks: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []
        
    print(f"Found SSIDs: {ssid_list}")
    ssid_list = sorted(list(set(ssid_list)))
    return ssid_list

def id_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

wpa_conf = """country=IN
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
network={
    ssid="%s"
    %s
}"""

wpa_conf_default = """country=IN
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
"""


PI_ID_FILE = '/home/dev/pi.id'
@app.route('/')
def main():
    piid = open(PI_ID_FILE, 'r').read().strip()
    return render_template('index.html', ssids=getssid(), message="Once connected you'll find IP address @ <a href='https://snaptext.live/{}' target='_blank'>snaptext.live/{}</a>.".format(piid,piid))

# Captive portal when connected with iOS or Android
@app.route('/generate_204')
def redirect204():
    return redirect("http://192.168.4.1", code=302)

@app.route('/hotspot-detect.html')
def applecaptive():
    return redirect("http://192.168.4.1", code=302)



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

def wifi_prerequisites():
    if not check_required_packages():
        return False
        
    if not check_wifi_interface():
        return False
    
    return True

def setup_ap():
    # things to run the first time it boots
    if not os.path.isfile(PI_ID_FILE):
        with open(PI_ID_FILE, 'w') as f:
            f.write(id_generator())
            
        # Use absolute path and wait for completion
        setup_script = os.path.join(currentdir, 'turnkey', 'setup_ap.sh')
        try:
            # Make script executable
            subprocess.run(['sudo', 'chmod', '+x', setup_script], check=True)
            
            # Run setup script and wait for completion
            result = subprocess.run(['sudo', setup_script], 
                                  check=True,
                                  capture_output=True,
                                  text=True)
                                  
            print(result.stdout)  # Print output for debugging
            if result.returncode == 0:
                print("AP setup completed successfully")
                time.sleep(2)  # Short wait after setup
                return True
            else:
                print(f"AP setup failed: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"Error running setup script: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error in setup_ap: {e}")
            return False
            
    piid = open(PI_ID_FILE, 'r').read().strip()
    print(piid)
    return True

async def connect_wifi():
    print("Checking wifi prerequisites")
    if not wifi_prerequisites():
        return False
    
    print("Loading wifi config")
    wifi_config = load_wifi_config()
    if not wifi_config:
        return False
    
    print("Cleaning up wireless interfaces")
    # Clean up before starting
    if not cleanup_wifi():
        print("Failed to cleanup wireless interfaces")
        return False
    
    try:
        for net in wifi_config['known_networks']:
            print(f"\nTrying to connect to {net['ssid']}   {net['password']}")
            
            # Create network configuratio
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
            
            print("Starting monitor wifi connection")
            status = await monitor_wifi_connection(timeout=10)
            if status:
                print(f"Successfully connected to {net['ssid']}")
                return True
            else:
                print(f"Failed to connect to {net['ssid']}")
                
    except Exception as e:
        print(f"Connection failed: {e}")
        
    return False


async def start_wpa_supplicant():
    try:
        # Start wpa_supplicant in background
        process = await asyncio.create_subprocess_exec(
            'sudo', 'wpa_supplicant',
            '-B',                    # Run in background
            '-i', 'wlan0',
            '-c', '/etc/wpa_supplicant/wpa_supplicant.conf',
            '-P', '/var/run/wpa_supplicant.pid',
            '-D', 'nl80211,wext'
        )
        
        # Give it a moment to start
        await asyncio.sleep(1)
        
        # Check if process is running by checking the PID file
        try:
            with open('/var/run/wpa_supplicant.pid', 'r') as f:
                pid = int(f.read().strip())
                # Check if process exists
                subprocess.run(['kill', '-0', str(pid)], check=True)
                print("wpa_supplicant started successfully")
                return True
        except:
            print("wpa_supplicant failed to start properly")
            return False

    except Exception as e:
        print(f"Error starting wpa_supplicant: {e}")
        return False

async def stop_wpa_supplicant():
    subprocess.run(['sudo', 'killall', 'wpa_supplicant'], check=False)
    await asyncio.sleep(1)
    return True

async def monitor_wifi_connection(timeout: int = 45):
    if not await start_wpa_supplicant():
        return False
    
    # Now monitor the connection status
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(['wpa_cli', 'status'], 
                                  capture_output=True, 
                                  text=True, 
                                  check=False)
            print(f"wpa_cli status: {result.stdout}")
            if 'wpa_state=COMPLETED' in result.stdout:
                print("Successfully connected to WiFi")
                return True
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error checking status: {e}")
    
    await stop_wpa_supplicant()
    print("Connection timed out")
    return False

async def create_ap(enable: bool = True):
    
    setup_ap()
    
    process = await asyncio.create_subprocess_exec(
        'sudo', 'sh', '-c', '/home/dev/src/firmware/turnkey/enable_ap.sh ' + ('enable' if enable else 'disable'),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Read output streams asynchronously
    while True:
        # Read stdout
        stdout_line = await process.stdout.readline()
        if stdout_line:
            line = stdout_line.decode().strip()
            print(f"AP setup: {line}")
            if line == ("Access Point " + ("enabled" if enable else "disabled")):
                print("AP created successfully")
                return True
        
        # Read stderr
        stderr_line = await process.stderr.readline()
        if stderr_line:
            print(f"AP error: {stderr_line.decode().strip()}")
            
        # Check if process has finished
        if process.stdout.at_eof() and process.stderr.at_eof():
            break
            
        await asyncio.sleep(0.1)
        
    # Wait for process to complete
    await process.wait()
    
    print("Failed to create AP")
    return False


async def main():
    if not await connect_wifi():
        print("Could not connect to any known networks")
        if await create_ap(enable=True):
            app.run(host="0.0.0.0", port=80, threaded=True)

if __name__ == '__main__':
    asyncio.run(main())
