# ⚙️ visual-tracking-control {#mainpage}

# Overview {#overview}

The **visual-tracking-control** project is a _suite_ of cross-platform applications for visual tracking and visual servoing for the humanoid robot platform iCub.

The suite includes:
 - **hand-tracking**: a visual end-effector tracker using a 3D model-aided particle filter [[1]](https://arxiv.org/abs/1703.04771);
 - **visual-servoing**: a visual servoing `YARP` plugin to control the pose (position and orientation) of the end-effector using image feedback [[2]](https://arxiv.org/abs/1710.04465).
   - The `visual-servoing` consist of two plugins, a client and server version. On one hand, the server must be always running via `yarpdev` and is responsible to command the robot. On the other hand, the client is dynamically loaded by any application using visual servoing interfaces and has the sole purpose of communicating commands to the server. This architecture enable distributed computation over different machine and decouples client and server implementations.

The output of the hand-tracking application can be visualized by means of the [iCubProprioception](https://github.com/claudiofantacci/iCubProprioception) module. iCubProprioception provides an augmented-reality application which superimposes the 3D model of the end-effector onto the camera images, using the estimated pose provided by hand-tracking.

[![visual-tracking-control home](https://img.shields.io/badge/BayesFilters-Home%20%26%20Doc-E0C57F.svg?style=flat-square)](https://robotology.github.io/visual-tracking-control/doxygen/doc/html/index.html)

[TOC]


# 🎛 Dependencies {#dependencies}

---
visual-tracking-control suite depends on
 - [BayesFilters](https://github.com/robotology/bayes-filters-lib) - `version >= 0.6`
 - [iCub](https://github.com/robotology/icub-main)
 - [iCubContrib](https://github.com/robotology/icub-contrib-common)
 - [OpenCV](http://opencv.org) - `version >= 3.3`, built with `CUDA >= 8.0`
 - [SuperimposeMesh](https://github.com/robotology/superimpose-mesh-lib) - `version >= 0.9`
 - [YARP](http://www.yarp.it)


# 🔨 Build the suite {#build-the-suite}

---
Use the following commands to build, install and link the library.

## Build {#build}
With `make` facilities:
```bash
$ git clone https://github.com/robotology/visual-tracking-control
$ cd visual-tracking-control
$ mkdir build && cd build
$ cmake -DBUILD_HAND_TRACKING=ON -DBUILD_VISUAL_SERVOING_CLIENT=ON -DBUILD_VISUAL_SERVOING_SERVER=ON ..
$ make
$ [sudo] make install
```

With IDE build tool facilities:
```bash
$ git clone https://github.com/robotology/visual-tracking-control
$ cd visual-tracking-control
$ mkdir build && cd build
$ cmake -DBUILD_HAND_TRACKING=ON -DBUILD_VISUAL_SERVOING_CLIENT=ON -DBUILD_VISUAL_SERVOING_SERVER=ON ..
$ cmake --build . --target ALL_BUILD --config Release
$ cmake --build . --target INSTALL --config Release
```


# 📑 References {#references}

---

 [1] C. Fantacci, U. Pattacini, V. Tikhanoff and L. Natale, "Visual end-effector tracking using a 3D model-aided particle filter for humanoid robot platforms", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Vancouver, BC, Canada, 2017. _arXiv preprint [arXiv:1703.04771](https://arxiv.org/abs/1703.04771)_.<br>
 [2] C. Fantacci, G. Vezzani, U. Pattacini, V. Tikhanoff and L. Natale, "Markerless visual servoing on unknown objects for humanoid robot platforms", IEEE International Conference on Robotics and Automation (ICRA), Brisbane, AU, 2018. _arXiv preprint [arXiv:1710.04465](https://arxiv.org/abs/1710.04465)_.
