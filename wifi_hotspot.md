# Install wireless tools if not already installed

For connecting to existing WIFI

sudo apt-get install dhcpcd5
sudo systemctl enable dhcpcd
sudo systemctl start dhcpcd


sudo hostapd -dd /etc/hostapd/hostapd.conf
For running the service in debug mode

Logs 
sudo systemctl stop hostapd

