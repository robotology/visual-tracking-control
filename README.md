# ⚙️ visual-tracking-control

The **visual-tracking-control** project is a _suite_ of cross-platform applications for visual tracking and visual servoing for the humanoid robot platform iCub.

The suite includes:
 - **hand-tracking**: a visual end-effector tracker using a 3D model-aided particle filter [[1]](https://arxiv.org/abs/1703.04771);
 - **visual-servoing**: a visual servoing `YARP` plugin to control the pose (position and orientation) of the end-effector using image feedback [[2]](https://arxiv.org/abs/1710.04465).
   - The `visual-servoing` consist of two plugins, a client and server version. On one hand, the server must be always running via `yarpdev` and is responsible to command the robot. On the other hand, the client is dynamically loaded by any application using visual servoing interfaces and has the sole purpose of communicating commands to the server. This architecture enable distributed computation over different machine and decouples client and server implementations.

The output of the hand-tracking application can be visualized by means of the [iCubProprioception](https://github.com/claudiofantacci/iCubProprioception) module. iCubProprioception provides an augmented-reality application which superimposes the 3D model of the end-effector onto the camera images, using the estimated pose provided by hand-tracking.

[![visual-tracking-control home](https://img.shields.io/badge/Visual%20Tracking%20Control-Home%20%26%20Doc-E0C57F.svg?style=flat-square)](https://robotology.github.io/visual-tracking-control/doxygen/doc/html/index.html) [![ZenHub](https://img.shields.io/badge/Shipping_faster_with-ZenHub-blue.svg?style=flat-square)](https://zenhub.com)


# Overview
- [🎛 Dependencies](#-dependencies)
- [🔨 Build the suite](#-build-the-suite)
- [📝 API documentation and example code](#-api-documentaion-and-example-code)
- [📑 Reference](#-reference)


# 🎛 Dependencies
visual-tracking-control suite depends on
 - [BayesFilters](https://github.com/robotology/bayes-filters-lib) - `version >= 0.9`
 - [iCub](https://github.com/robotology/icub-main)
 - [iCubContrib](https://github.com/robotology/icub-contrib-common)
 - [OpenCV](http://opencv.org) - `version >= 3.3`, built with `CUDA >= 8.0`
 - [SuperimposeMesh](https://github.com/robotology/superimpose-mesh-lib) - `version >= 0.9`
 - [YARP](http://www.yarp.it)


# 🔨 Build the suite
Use the following commands to build, install and link the library.

### Build
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

# 📝 API documentaion and example code
Doxygen-generated documentation is available [here](https://robotology.github.io/visual-tracking-control/doxygen/doc/html/index.html).


## 📑 References

[1] C. Fantacci, U. Pattacini, V. Tikhanoff and L. Natale, "Visual end-effector tracking using a 3D model-aided particle filter for humanoid robot platforms", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Vancouver, BC, Canada, 2017. _arXiv preprint [arXiv:1703.04771](https://arxiv.org/abs/1703.04771)_.  
[2] C. Fantacci, G. Vezzani, U. Pattacini, V. Tikhanoff and L. Natale, "Markerless visual servoing on unknown objects for humanoid robot platforms", IEEE International Conference on Robotics and Automation (ICRA), Brisbane, AU, 2018. _arXiv preprint [arXiv:1710.04465](https://arxiv.org/abs/1710.04465)_.

---
[![how-to-export-cpp-library](https://img.shields.io/badge/-Project%20Template-brightgreen.svg?style=flat&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAEAAAAA9CAYAAAAd1W%2FBAAAABmJLR0QA%2FwD%2FAP%2BgvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH4QEFECsmoylg4QAABRdJREFUaN7tmmuIVVUUx%2F%2F7OmpaaGP6oedkGJWNIWoFVqRZGkIPSrAQgqhEqSYxszeFUB%2FCAqcXUaSRZmZP6IFm42QEUWAjqT1EQ0dLHTMfaWajv76sM%2BxO59znuY%2Bcs2CYmXv33mud31577bX3WU5lEEDOueDvfpLGSBolaaiksyUNknRyqNs%2BSR2SfrKf1ZJaJG11zv1rzJoX4ETgYWAtpcuvwCvABQHcJMUlPevAi5KmxTTbKalN0hZJ2yRlvO%2BOlzTYvOScmP5fSrreOber1mZcQF9gU2j2dgDNwLgixmwE7ge%2BC415FDi%2FFt1%2BuWfkRuBqH1CJYw8B3vfG7wR61NLDn%2BoZt6IcHma%2F7%2FX0zEo6HpRi4KWeYWOTNswfz9OzoKpr3ov2s4HNnmHtwMAy6Vvk6VkPjKkWgInA5zm2r0eBulJn3P6%2FEdgZo2c%2F8BDQu9wP3Qg8DRyIMGJPFhCfAjOAUcAgwOXQ08%2BC3hSb8SMF5AyfANcG4Iteip7L9QMejNjeAlkEjLZ1n490Ah023g%2FAZ0AL8DWwAdgO%2FBnT9y%2Fgdm8CllggbI9ouxeYD4wsNtBcBXwcY8hGYGqo7xjKJyuAyZ6uQ%2Fb5fO%2BzEcCbMf23ANNzeZ6AYcA8oxeWbcDcIAGJWKOlANgCfGNesBR4Cpjqz15ocgIAr0Z4bE%2FgDhsvSt71kzJAAm7O4uJvABfnSmhKBNBY4PL8D4CYdqcBc4CDETp%2Fs3g2SDFGNRVoVCkARhQYlwJ5vgD7JgDLInTvzsT0mQd8BFyTTzrrnGstd84hqR5Y5321LJtNHrABks6V1FfSkVCzeuUxQweAl4Ah2WAAd5XDA4AzgOdCfVbmAe4G22GI2SXATnGFyBrg1rikw05vhcpwIGMBrI%2Bt3UnAMxYgw7Lc7I7Sf7oF0ajcYZ%2BdTBuA24oF4O%2FnS4ErI4w4E3irgLF22f5%2FMEe7r4AJ3vG7y8WBO4Fvs0T%2B8SEb7y4VgC%2B%2FW0QdGFLSC5hmsaRYWWNp7ikRoK%2FL4uLrbZZ7xnhqFwBHske3lZKelfSBc%2B5o6G6wQdJIuxMcIKnBu5FykrZL2iVpq6TVzrm2CMMHS5ouaYak8MPtlfS6pGbn3Ibw3WQYgKTm8LaSpOwHFgCXJHAC7A80AW0xupb4SzGf%2BUx6CeSzxmcBmQLT8Yl2VoiSDZbx9SgSbkUB%2BPKeHZwyMSn1YOBJ4HBM9tYMnFfqNVs1AQTSYQ8zDOgN3AOsi2n7jn%2FxkUTIqgUAuWSTbW3lyi67ANSpdmS3pIWSXnbOra2U0loB8IikJ4JXYJWUTI0AaA%2F260q%2F%2F8uom0sKIAWQAkgBpABSACmAFEAKIAWQAkgBpABSACmAFEB3kc5uBSD0wuUySVN8AB3dgEF%2FK7PdLWmVpOCV3dGMpCGSZkr6%2FliabeA44CagVdIeSXMl1XtNV0kaH%2B58VkQ1RiXklgQBjAYWW11hVLXbfVY2k3OgKfZ%2BvuYB2Bvk2THltIetYOOiYl2pAXgM%2BLkWAHh21dkktcaM2WolgD3DgbCUCDoceK3KAC7MUkO8A5gJ1Fci2DQBP1YCAHCSFWD9EtH3b3Pxy6sVdYdaZVZHEgA8Fw%2Fi0BcxfVqAyUCvklw84STjCuDDEgEMBxbGtPsDeAA4odb34D5WZt%2BeJ4AmK6PZHPHdQeBtYOz%2FNTEZCbwQU%2FaSq0x%2BEtCnqi6eMIxJWUrZAxd%2FPHjoY%2FZQYrnFHIvqh2zNj6uGTf8ARTOPo64fR94AAAAASUVORK5CYII%3D)](https://github.com/robotology/how-to-export-cpp-library)
