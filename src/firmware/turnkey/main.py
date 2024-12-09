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
from flask import Flask, render_template, redirect, request

# Use absolute paths
BASE_DIR = '/home/dev/src/firmware/turnkey'
template_dir = os.path.join(BASE_DIR, 'templates')
static_dir = os.path.join(BASE_DIR, 'static')

app = Flask(__name__,
            static_url_path='',
            static_folder=static_dir,
            template_folder=template_dir)

ssid_list = []
def getssid():
    global ssid_list
    if len(ssid_list) > 0:
        return ssid_list
    ssid_list = []
    
    try:
        subprocess.run(['sudo', 'killall', 'wpa_supplicant'], check=False)  # Clean up any existing instances
        time.sleep(1)

        # Start wpa_supplicant with explicit control interface
        subprocess.run([
            'sudo', 'wpa_supplicant',
            '-B',                    # Run in background
            '-i', 'wlan0',          # Interface name
            '-c', '/etc/wpa_supplicant/wpa_supplicant.conf',
            '-D', 'nl80211,wext',
            '-P', '/var/run/wpa_supplicant.pid'
        ], check=True)
        
        time.sleep(2)  # Give it time to initialize
        # Initiate scan
        subprocess.run(['sudo', 'wpa_cli', 'scan'], check=True)
        time.sleep(2)  # Wait for scan to complete
        
        # Get scan results
        scan_results = subprocess.check_output(['sudo', 'wpa_cli', 'scan_results'], 
                                             text=True)
        
        # Parse results
        print(scan_results)
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

@app.route('/restart', methods=['POST', 'GET'])
def restart():
    raise Exception("Not implemented")

@app.route('/signin', methods=['POST'])
def signin():
    try: 
        print ("In the signing post")
        email = request.form['email'] 
        password = request.form['password'] or "11223344"
        ssid = request.form['ssid'] or "Excitel-5G"
        key_mgmt = request.form['key_mgmt'] or 'WPA-PSK'
        print(f"Connecting to {ssid} with {key_mgmt} security")
        # Create network configuration
        network_config = (
            'ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev\n'
            'update_config=1\n'
            'country=US\n\n'
            f'network={{\n'
            f'    ssid="{ssid}"\n'
            f'    psk="{password}"\n'
            f'    key_mgmt={key_mgmt}\n'
            f'}}\n'
        )

        # Write configuration
        with open('/etc/wpa_supplicant/wpa_supplicant.conf', 'w') as f:
            f.write(network_config)
        
        return "Successfully updated WiFi configuration. It should connect automatically."
    except Exception as e:
        return f"Error: {e}", 500


# def enable_services():
#     try:
#         # Enable wpa_supplicant first
#         subprocess.run(['sudo', 'systemctl', 'enable', 'wpa_supplicant'], check=True)
        
#         # Check if dhcpcd service exists before enabling
#         result = subprocess.run(['systemctl', 'list-unit-files', 'dhcpcd.service'], 
#                               capture_output=True, 
#                               text=True)
        
#         if 'dhcpcd.service' in result.stdout:
#             subprocess.run(['sudo', 'systemctl', 'enable', 'dhcpcd'], check=True)
#         else:
#             print("dhcpcd service not found, installing...")
#             subprocess.run(['sudo', 'apt-get', 'update'], check=True)
#             subprocess.run(['sudo', 'apt-get', 'install', '-y', 'dhcpcd5'], check=True)
#             subprocess.run(['sudo', 'systemctl', 'enable', 'dhcpcd'], check=True)
            
#     except subprocess.CalledProcessError as e:
#         print(f"Error enabling services: {e}")
#         return False
#     return True

# def check_required_packages():
#     enable_services()
#     required_packages = ['wireless-tools', 'wpasupplicant']
#     missing_packages = []
    
#     try:
#         # First check which packages are missing
#         for package in required_packages:
#             result = subprocess.run(['dpkg', '-s', package], 
#                                   capture_output=True, 
#                                   text=True)
#             if result.returncode != 0:
#                 missing_packages.append(package)
        
#         # Only update and install if there are missing packages
#         if missing_packages:
#             print(f"Missing packages: {missing_packages}")
#             print("Updating package list...")
#             subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            
#             for package in missing_packages:
#                 print(f"Installing {package}...")
#                 subprocess.run(['sudo', 'apt-get', 'install', '-y', package], check=True)
        
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to install required packages: {e}")
#         return False

# def check_wifi_interface():
#     try:
#         # Check if wlan0 exists
#         result = subprocess.run(['ip', 'link', 'show', 'wlan0'], capture_output=True, text=True)
#         if result.returncode != 0:
#             print("WiFi interface wlan0 not found!")
#             return False
            
#         # Ensure WiFi is not blocked
#         subprocess.run(['sudo', 'rfkill', 'unblock', 'wifi'], check=False)
        
#         # Bring interface up
#         subprocess.run(['sudo', 'ip', 'link', 'set', 'wlan0', 'up'], check=True)
#         time.sleep(2)  # Give interface time to come up
        
#         return True
#     except Exception as e:
#         print(f"Error checking WiFi interface: {e}")
#         return False

# def cleanup_wifi():
#     """Thoroughly clean up all wireless interfaces and processes"""
#     try:
#         # Stop all related services
#         services = ['wpa_supplicant', 'dhcpcd', 'NetworkManager', 'dhclient']
#         for service in services:
#             subprocess.run(['sudo', 'systemctl', 'stop', service], check=False)
#             subprocess.run(['sudo', 'killall', service], check=False)
        
#         # Remove the problematic control interface file
#         subprocess.run(['sudo', 'rm', '-rf', '/var/run/wpa_supplicant'], check=False)
#         subprocess.run(['sudo', 'mkdir', '-p', '/var/run/wpa_supplicant'], check=False)
        
#         # Reset the wireless interface
#         subprocess.run(['sudo', 'ip', 'link', 'set', 'wlan0', 'down'], check=False)
#         time.sleep(1)
#         subprocess.run(['sudo', 'ip', 'link', 'set', 'wlan0', 'up'], check=False)
#         time.sleep(2)
        
#         return True
#     except Exception as e:
#         print(f"Cleanup failed: {e}")
#         return False

# def wifi_prerequisites():
#     if not check_required_packages():
#         return False
        
#     if not check_wifi_interface():
#         return False
    
#     return True

def kill_another_instance():
    # kill a process running on the same port 80
    try:
        # Find process using port 80
        result = subprocess.check_output(['sudo', 'lsof', '-t', '-i:80'], text=True)
        pids = result.strip().split('\n')
        
        # Kill each process found
        for pid in pids:
            if pid:  # Check if pid is not empty
                try:
                    # Kill process
                    subprocess.run(['sudo', 'kill', '-9', pid.strip()], check=True)
                    print(f"Killed process {pid} running on port 80")
                except subprocess.CalledProcessError:
                    print(f"Failed to kill process {pid}")
        return True
        
    except subprocess.CalledProcessError:
        # No process found on port 80
        print("No process found running on port 80")
        return True
    except Exception as e:
        print(f"Error killing process on port 80: {e}")
        return False
    # kill a process running on the same port 80
    

    
def setup_ap():
    # things to run the first time it boots
    if not os.path.isfile(PI_ID_FILE):
        with open(PI_ID_FILE, 'w') as f:
            f.write(id_generator())
            
        # Use absolute path and wait for completion
        setup_script = os.path.join(BASE_DIR, 'setup_ap.sh')
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

async def start_wpa_supplicant():
    try:
        # Check if wpa_supplicant is already running
        try:
            with open('/var/run/wpa_supplicant.pid', 'r') as f:
                pid = int(f.read().strip())
                # Check if process exists
                subprocess.run(['kill', '-0', str(pid)], check=True)
                print("wpa_supplicant already running")
                return True
        except:
            # Not running, so start it
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
            
            # Verify it started successfully
            try:
                with open('/var/run/wpa_supplicant.pid', 'r') as f:
                    pid = int(f.read().strip())
                    subprocess.run(['kill', '-0', str(pid)], check=True)
                    print("wpa_supplicant started successfully")
                    return True
            except:
                print("wpa_supplicant failed to start properly")
                return False

    except Exception as e:
        print(f"Error with wpa_supplicant: {e}")
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
    kill_another_instance()
    if not await monitor_wifi_connection(timeout=10):
        print("Could not connect to any known networks")
        if await create_ap(enable=True):
            app.run(host="0.0.0.0", port=80, threaded=True)

if __name__ == '__main__':
    asyncio.run(main())
