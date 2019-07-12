#!/bin/sh
#
# \description  Install requirements
# Note: this script is designed for Linux Ubuntu and might also work on Debian
# or other Linuxes
#
# \author Artem V L <artem@exascale.info>  https://exascale.info

# Update packages information
sudo apt-get -y update
# Get errorcode of the last operation and terminate if the update failed
ERR=$?
if [ $ERR -ne 0 ]
then
	echo "ERROR, the dependencies installation terminated, \"apt-get update\" failed with the code $ERR"
	exit $ERR
fi

# Note: "free" (not an apt package) and "sed" are typically installed by default
sudo apt-get install -y \
	python3 python3-pip \
	sed bc \
	parallel

# Check and set locale if required
if [ "$LC_ALL" = '' ]
then
	export LC_ALL="en_US.UTF-8"
	export LC_CTYPE="en_US.UTF-8"
fi

# Note: Python3 and pip3 were installed on previous step
sudo pip3 install --upgrade pip

# Install Python dependencies
sudo pip3 install -r requirements.txt

# Build the Cython lib
./build.sh
