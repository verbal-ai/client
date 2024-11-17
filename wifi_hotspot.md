# Install wireless tools if not already installed

For connecting to existing WIFI

sudo apt-get install dhcpcd5
sudo systemctl enable dhcpcd
sudo systemctl start dhcpcd
