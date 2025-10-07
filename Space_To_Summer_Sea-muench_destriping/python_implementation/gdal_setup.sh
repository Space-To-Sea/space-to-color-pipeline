# Use this if you are using ubuntu for post processing

# Install dependencies
sudo apt update
sudo apt install build-essential cmake wget git
sudo apt install libproj-dev libgeos-dev libsqlite3-dev sqlite3
sudo apt install libtiff-dev libcurl4-openssl-dev libxml2-dev

# Download and compile GDAL 3.11.0
wget https://github.com/OSGeo/gdal/releases/download/v3.11.0/gdal-3.11.0.tar.gz
tar -xzf gdal-3.11.0.tar.gz
cd gdal-3.11.0

# Configure and build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install

# Update library path
sudo ldconfig
