#!/bin/bash

DEB_ARCH=$(dpkg --print-architecture)
DEB_BUILD_ROOT=$(pwd)/build

DEB_PKG_VERSION=$(grep '^version' $(pwd)/Cargo.toml | head -1 | cut -d'"' -f2)
DEB_PKG_NAME=xng-${DEB_PKG_VERSION}-${DEB_ARCH}

mkdir -p ${DEB_BUILD_ROOT}/${DEB_PKG_NAME}/DEBIAN
mkdir -p ${DEB_BUILD_ROOT}/${DEB_PKG_NAME}/usr/bin

cp $(pwd)/target/release/xng ${DEB_BUILD_ROOT}/${DEB_PKG_NAME}/usr/bin/xng
mkdir -p ${DEB_BUILD_ROOT}/${DEB_PKG_NAME}/usr/lib/udev/rules.d
cp $(pwd)/packaging/60-xng-airspy.rules ${DEB_BUILD_ROOT}/${DEB_PKG_NAME}/usr/lib/udev/rules.d/
sed -e s/CURRENT_ARCH/${DEB_ARCH}/ -e s/CURRENT_VERSION/${DEB_PKG_VERSION}/ $(pwd)/packaging/control > ${DEB_BUILD_ROOT}/${DEB_PKG_NAME}/DEBIAN/control

echo ${DEB_PKG_VERSION} > ${DEB_BUILD_ROOT}/version

pushd ${DEB_BUILD_ROOT} && \
  dpkg-deb --build ${DEB_PKG_NAME} && \
  popd

