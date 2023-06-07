#!/bin/bash

# color definitions
CYAN='\033[0;36m'
NC='\033[0m'

# exit if any command fails
set -e

SCRIPT_PATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
INSTALL_PREFIX="${SCRIPT_PATH}/ext"

echo -e "${RED} Installing to ${SCRIPT_PATH}/ext"

git submodule init
git submodule update

# ---- build and install blis to the ext folder ----
build_install_blis() {
    cd third_party/blis

    echo -e "${RED}~~~~~ Building BLIS ~~~~~${NC}"

    ./configure --prefix=${INSTALL_PREFIX} --enable-threading=single --enable-cblas $(uname -i)
    make -j
    make check

    echo -e "${RED}~~~~~ Installing BLIS ~~~~~${NC}"
    make install

    cd ${SCRIPT_PATH}
}

build_install_eigen() {
    cd third_party/Eigen
    
    echo -e "${RED}~~~~~ Building Eigen ~~~~~${NC}"

    mkdir build/
    cd build/

    cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} ../
    make install

    cd ${SCRIPT_PATH}

}

build_install_fftw() {
    cd third_party/

    rm -rf fftw-3.3.10/
    wget http://fftw.org/fftw-3.3.10.tar.gz
    tar -zxf fftw-3.3.10.tar.gz
    rm -rf fftw-3.3.10.tar.gz

    cd fftw-3.3.10/
    # enabled vector instructions for my particular machine
    ./configure CFLAGS="-march=native -O3" --prefix=${INSTALL_PREFIX} --enable-sse2 --enable-avx --enable-avx2 
    
    make -j
    make install

    cd ${SCRIPT_PATH}

}

# ---- build and install to the ext folder ----
# build_install_blis
# build_install_eigen
build_install_fftw