#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.



"""
Extract background images from a tar archive.

"""


__all__ = (
    'extract_backgrounds',
)


import os
import sys
import tarfile

import cv2
import numpy
import shutil
import xml.etree.ElementTree as et

def get_rect(f):
    tree=et.parse(f)
    e=tree.find("object")
    e=e.find("bndbox")
    xmin=e.find("xmin")
    ymin=e.find("ymin")
    xmax=e.find("xmax")
    ymax=e.find("ymax")
    return xmin.text,ymin.text,xmax.text,ymax.text

def extract_backgrounds(picdir):
    """
    Extract backgrounds from provided tar archive.

    JPEGs from the archive are converted into grayscale, and cropped/resized to
    256x256, and saved in ./bgs/.

    :param archive_name:
        Name of the .tar file containing JPEGs of background images.

    """
    if not os.path.exists(picdir):
        print "{} doesn't exist.".format(picdir)
        return
    if os.path.exists("gray"):
        shutil.rmtree("gray")
    os.mkdir("gray")

    for parent,dirnames,filenames in os.walk(picdir):
        for filename in filenames:
            fname=os.path.join("pic",filename)
            img=cv2.imread(fname,cv2.CV_LOAD_IMAGE_GRAYSCALE)
            fname="{}.xml".format(filename.split('.')[0])
            x1,y1,x2,y2=get_rect(os.path.join("data",fname))
            roi=img[y1:y2, x1:x2]
            #cv2.getRectSubPix()
            fname=os.path.join("gray",filename)
            rc=cv2.imwrite(fname,roi)
            if not rc:
                print "fail to write to file {}".format(fname)

"""
    def members():
        m = t.next()
        while m:
            yield m
            m = t.next()
    index = 0
    for m in members():
        if not m.name.endswith(".jpg"):
            continue
        f =  t.extractfile(m)
        try:
            im = im_from_file(f)
        finally:
            f.close()
        if im is None:
            continue
        
        if im.shape[0] > im.shape[1]:
            im = im[:im.shape[1], :]
        else:
            im = im[:, :im.shape[0]]
        if im.shape[0] > 256:
            im = cv2.resize(im, (256, 256))
        fname = "bgs/{:08}.jpg".format(index)
        print fname
        rc = cv2.imwrite(fname, im)
        if not rc:
            raise Exception("Failed to write file {}".format(fname))
        index += 1
"""
"""
usage: extracgbgs  $source_picture_directory
"""
if __name__ == "__main__":

    extract_backgrounds(sys.argv[1])

