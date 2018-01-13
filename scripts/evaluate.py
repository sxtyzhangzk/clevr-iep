import argparse
import json
import random
import shutil
import sys
import glob
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter


def changeColor(x):
    if (abs(x-0.25) < 0.0001): return 137
    if (abs(x-0.5) < 0.0001): return 188
    if (abs(x-0.75) < 0.0001): return 225
    return 0


def normalize(a):
    W = len(a)
    H = len(a[0])
    sum = 0
    for x in range(0, W):
        for y in range(0, H):
            sum += a[x][y]
    if (sum == 0) : return a
    for x in range(0, W):
        for y in range(0, H):
            if (a[x][y] > 0) :
                a[x][y] = float(a[x][y]) / sum
    return a


def do_evaluate(segment_map, focus_name, answer_object, scene_name):

    focus_map = Image.open(focus_name).convert("L")
    #focus_map.show()
    W, H = focus_map.size
    #print((W,H))
    fm = [[0.0 for x in range(H)] for y in range(W)]
    maxl=0
    for x in range(0, W):
        for y in range(0, H):
            l = focus_map.getpixel((x, y))
            if(maxl<l):maxl=l
            fm[x][y] = l
    for x in range(0, W):
        for y in range(0, H):
            fm[x][y]/=maxl
    #fm = normalize(fm)

    seg = Image.open(segment_map)
    #seg.show()
    scene_file = open(scene_name, encoding='utf-8')
    scene = json.load(scene_file)
    scene_file.close()
    idx = 0
    res = []
    for obj in scene['objects']:
        ground_truth = (idx in answer_object)
        idx += 1
        (r, g, b) = obj['seg_color']
        R = changeColor(r)
        G = changeColor(g)
        B = changeColor(b)
        mask = Image.new("L", seg.size, "black")
        draw = ImageDraw.Draw(mask)
        for x in range(W):
            for y in range(H):
                color = seg.getpixel((x, y))
                if (color == (R, G, B, 255)):
                    draw.point((x, y), (255))
                else:
                    draw.point((x, y), (0))
        mask = mask.filter(ImageFilter.GaussianBlur(3))
        weight = [[0.0 for x in range(H)] for y in range(W)]
        for x in range(W):
            for y in range(H):
                weight[x][y] = mask.getpixel((x, y))
        weight = normalize(weight)
        score = 0.0
        for x in range(W):
            for y in range(H):
                score += weight[x][y] * fm[x][y]
        res.append((score, ground_truth))
    return res


for filename in glob.glob("output/*.seg.png"):
    #print(filename)
    result = do_evaluate(filename, filename.replace('.seg', '.focus'), [0, 3, 4], filename.replace('.seg.png', '.json'))
    print(result)
