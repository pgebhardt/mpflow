# fastEIT

fastEIT is an extensible GPU-based Impedance Tomography Solving Library. It uses CUDA 5.0 and C++11.

[![CI](http://c64.est.ruhr-uni-bochum.de/projects/1/status?ref=master)](http://c64.est.ruhr-uni-bochum.de/projects/1?ref=master)

## Features

* 2.5D based forward solver
* linear and non-linear inverse solving
* independent of numerical algorithms (CG is shiped)
* high performance and proven accuracy

## System Requirements

* NVIDIA CUDA ToolKit 5.0
* clang 3.1
* scons 2.2.0

It has been tested on the following platforms:

* OS X 10.7
* Ubuntu Linux 12.04 x64
* Fedora Linux 18 x64

## Install

To build and install fastEIT a Scons script is included. It is designed to work on both Linux and OS X.

    scons
    scons install