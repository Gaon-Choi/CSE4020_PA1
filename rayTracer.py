#!/usr/bin/env python3
# -*- coding: utf-8 -*
# sample_python aims to allow seamless integration with lua.
# see examples below

import os
import sys
import pdb  # use pdb.set_trace() for debugging
import code # or use code.interact(local=dict(globals(), **locals()))  for debugging.
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

class Color:
    def __init__(self, R, G, B):
        # R: red, G: green, B: blue
        self.color=np.array([R,G,B]).astype(float)

    # Gamma corrects this color.
    # @param gamma the gamma value to use (2.2 is generally used).
    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma
        self.color=np.power(self.color, inverseGamma)

    def toUINT8(self):
        return (np.clip(self.color, 0,1)*255).astype(np.uint8)

class Light:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity

class Sphere:
    def __init(self, shader, radius, center):
        self.shader = shader
        self.radius = radius
        self.center = center

class Box:
    def __init__(self, shader_, minPt_, maxPt_, normals_):
        self.shader = shader_
        self.minPt = minPt_     # 3d point(minimum)
        self.maxPt = maxPt_     # 3d point(maximum)
        self.normals = normals_ # normal vectors, total 6

class Shader:
    def __init__(self, name_, type_):
        self.name = name_
        self.type = type_  # Phong or Lambertian

class Phong(Shader):    # child class_2 of Shader
    def __init__(self, diffuseColor_):
        self.diffuseColor = diffuseColor_

class Lambertian(Shader):   # child class_1 of Shader
    def __init__(self, diffuseColor_, specularColor_, exponent_):
        self.diffuseColor = diffuseColor_
        self.specularColor = specularColor_
        self.exponent = exponent_


def hit_sphere(center, radius, ray):
    origin = np.array[0, 0, 0] - center
    a = np.dot(ray, ray)
    b = 2.0 * np.dot(origin, ray)
    c = np.dot(origin, origin) - radius ** 2

    discriminant = (b ** 2) - 4 * a * c

    return (discriminant > 0)   # true if the ray hits the surface!

def raytracing():


def main():
    tree = ET.parse(sys.argv[1])
    root = tree.getroot()

    # container for shapes, shaders, lights
    shapes = list()
    shaders = list()
    lights = list()

    # set default values
    viewDir = np.array([0, 0, -1]).astype(float)
    viewUp = np.array([0, 1, 0]).astype(float)
    viewProjNormal = -1 * viewDir  # you can safely assume this. (no examples will use shifted perspective camera)
    viewWidth = 1.0
    viewHeight = 1.0
    projDistance = 1.0
    intensity = np.array([1, 1, 1]).astype(float)  # how bright the light is.
    print(np.cross(viewDir, viewUp))

    imgSize = np.array(root.findtext('image').split()).astype(int)

    for c in root.findall("camera"):
        viewPoint = np.array(c.findtext('viewPoint').split()).astype(float)
        print('viewpoint', viewPoint)

    for c in root.findall("shader"):
        diffuseColor_c = np.array(c.findtext('diffuseColor').split()).astype(float)
        print('name', c.get("name"))
        print('diffuseColor', diffuseColor_c)

    for c in root.findall("light"):
        tmp_p = np.array((c.findtext("position")).split()).astype(np.float64)
        tmp_i = np.array((c.findtext("intensity")).split()).astype(np.float64)
        tmp_Light = Light(tmp_p, tmp_i)
        lights.append(tmp_Light)
        print(tmp_Light)


    # Create an empty image
    channels = 3
    img = np.zeros((imgSize[1], imgSize[0], channels), dtype=np.uint8)
    img[:,:] = 0
    
    # replace the code block below!
    for i in np.arange(imgSize[1]): 
        white = Color(1,1,1)    # white = red + green + blue
        red = Color(1,0,0)
        blue = Color(0,0,1)
        img[10][i] = white.toUINT8()
        img[i][i] = red.toUINT8()
        img[i][0] = blue.toUINT8()

    for x in np.arange(imgSize[0]): 
        img[5][x] = [255, 255, 255]

    for y in np.arange(imgSize[0]):
        for x in np.arange(imgSize[1]):
            # Compute ray-tracing
            img[y][x] = blue.toUINT8()

    rawimg = Image.fromarray(img, 'RGB')

    #rawimg.save('out.png')
    rawimg.save(sys.argv[1]+'.png')
    
if __name__ == "__main__":
    main()
