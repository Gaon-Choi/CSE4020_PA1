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
import math
from PIL import Image

INF = math.inf

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

    def intersects(self, ray):
        sphere_to_ray = ray.origin - self.center
        # a = 1
        b = 2 * np.sum(normalize(ray) * sphere_to_ray)
        c = np.sum(sphere_to_ray * sphere_to_ray) - self.radius * self.radius
        discriminant = b ** 2 - 4 * c

        if discriminant >= 0:
            dist =  (-b - np.sqrt(discriminant)) / 2
            if dist > 0:
                return dist
        return None

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

class Lambertian(Shader):    # child class_1 of Shader
    def __init__(self, diffuseColor_):
        self.diffuseColor = diffuseColor_

class Phong(Shader):   # child class_2 of Shader
    def __init__(self, diffuseColor_, specularColor_, exponent_):
        self.diffuseColor = diffuseColor_
        self.specularColor = specularColor_
        self.exponent = exponent_

class Camera:
    def __init__(self, viewPoint_, viewDir_, projNormal_, viewUp_, projDistance_, viewWidth_, viewHeight_):
        viewPoint = viewPoint_
        viewDir = viewDir_
        projNormal = projNormal_
        viewUp = viewUp_
        projDistance = projDistance_
        viewWidth = viewWidth_
        viewHeight = viewHeight_

def hit_sphere(a, b, c):
    discriminant = (b ** 2) - 4 * a * c
    return (discriminant > 0)   # true if the ray hits the surface!

def normalize(vec):
    # return unit vector which is parallel to vec
    # vec / |vec|
    return vec / np.sqrt(np.sum(vec * vec))

def raytracing(shapes, ray, viewPoint):
    closest = -1
    cnt = 0
    global INF
    d = INF
    for objects in shapes:
        if objects.__class__.__name__ == "Sphere":
            if objects.intersects(ray) != None:
                if d > objects.intersects(ray):
                    d = objects.intersects(ray)
                    closest = cnt
        cnt += 1
    return [d, closest]

def shade(d, ray, viewPoint, shapes, closest, light):
    if closest == -1:
        # no objects intersect
        return np.array(np.zeros(1, 3))
    else:
        x = 0; y = 0; z = 0
        n = np.array([0, 0, 0])
        v = -d * ray

        if list[closest].__class__.__name__ == "Sphere":
            n = viewPoint + d * ray - shapes[closest].center
            n = normalize(n)

        result_ = Color(x, y, z)
        result_.gammaCorrect(2.2)
        return result_.toUINT8()

def main():
    tree = ET.parse(sys.argv[1])
    root = tree.getroot()

    # container for shapes, shaders, lights
    shapes = list()
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

    # <scene> - <camera>
    for c in root.findall("camera"):
        if (c.findtext("viewPoint") == True):
            viewPoint = np.array(c.findtext('viewPoint').split()).astype(float)
        if (c.findtext("viewDir") == True):
            viewDir = np.array(c.findtext("viewDir").split()).astype(float)
        if (c.findtext("projNormal") == True):
            projNormal = np.array(c.findtext("projNormal").split()).astype(float)
        if (c.findtext("viewUp") == True):
            viewUp = np.array(c.findtext("viewUp").split()).astype(float)
        if(c.findtext("projDistance") == True):
            projDistance = np.array(c.findtext("projDistance").split()).astype(float)
        if(c.findtext("viewWidth") == True):
            viewWidth = np.array(c.findtext("viewWidth").split()).astype(float)
        if(c.findtext("viewHeight") == True):
            viewHeight = np.array(c.findtext("viewHeight").split()).astype(float)

    for c in root.findall("surface"):
        # Case 1 : Sphere
        if (c.get("type") == "Sphere"):
            center_ = np.array(c.findtext("center").split()).astype(float)
            radius_ = np.array(c.findtext("radius").split()).astype(float)

            ref = str()
            for d in c:
                if d.tag == "shader":
                    ref = d.get("ref")  # find matching btw. shader <-> surface

            all_shaders = root.findall("shader")
            for e in all_shaders:
                # only one shader!
                if e.get("name") == ref:
                    obj = object()  # Surface: sphere or box
                    diffuseColor_ = np.array(e.findtext("diffuseColor").split()).astype(float)
                    shader_ = object(); specularColor_ = object(); exponent_ = object()
                    if (e.get("type") == "Lambertian"):
                        shader_ = Lambertian(diffuseColor_)

                    elif (e.get("type") == "Phong"):
                        specularColor_ = np.array(e.findtext("specularColor").split()).astype(float)
                        exponent_ = np.array(e.findtext("exponent").split()).astype(float)
                        shader_ = Phong(diffuseColor_, specularColor_, exponent_)

                    obj = Sphere(shader_, radius_, center_)
                    shapes.append(obj)

        # Case 2 : Box
        elif(c.get("type") == "Box"):
            minPt_ = np.array(c.findtext("minPt").split()).astype(float)
            maxPt_ = np.array(c.findtext("maxPt").split()).astype(float)
            p = [minPt_, maxPt_]

            normal_vectors = list()
            points = list()

            ref = str()
            for d in c:
                if d.tag == "shader":
                    ref = d.get("ref")  # find matching btw. shader <-> surface

    for c in root.findall("light"):
        tmp_p = np.array((c.findtext("position")).split()).astype(np.float64)   # position
        tmp_i = np.array((c.findtext("intensity")).split()).astype(np.float64)  # intensity
        tmp_Light = Light(tmp_p, tmp_i)
        lights.append(tmp_Light)
        print("Light ", len(lights), " :", tmp_Light)


    # Create an empty image
    channels = 3
    img = np.zeros((imgSize[1], imgSize[0], channels), dtype=np.uint8)
    img[:,:] = 0

    pixel_x = viewWidth / imgSize[0]
    pixel_y = viewHeight / imgSize[1]

    w = viewDir
    u = np.cross(w, viewUp)
    v = np.cross(w, u)

    # unit vectors: w, u, v
    unit_w = normalize(w)
    unit_u = normalize(u)
    unit_v = normalize(v)

    start = unit_w * projDistance - unit_u * pixel_x * ((imgSize[0] / 2) + 1/2) - unit_v * pixel_y * ((imgSize[1] / 2) + 1/2)

    for y in np.arange(imgSize[0]):
        for x in np.arange(imgSize[1]):
            # Compute ray-tracing
            ray = start + unit_u * x * pixel_x + pixel_y * y * unit_v
            result = raytracing(shapes, ray, viewPoint)
            img[y][x] = result

    rawimg = Image.fromarray(img, 'RGB')

    #rawimg.save('out.png')
    rawimg.save(sys.argv[1]+'.png')

if __name__ == "__main__":
    main()
