#!/usr/bin/env python

__all__ = (

)

import sys
import os
import os.path
import xml.etree.ElementTree as et

def walkdir(rootdir):
    if not os.path.exists(rootdir):
        print "{} doesn't exist.".format(rootdir)
        return

    ret=[]
    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            tree=et.parse(os.path.join(parent,filename))
            root=tree.getroot()
            folderelement=tree.find("folder")
            filenameelement=tree.find("filename")
            pathelement=tree.find("path")
            #print "\nfolder:{} \nfilename: {}\npath: {}".format(folderelement.text,filenameelement.text,pathelement.text)
            #pathelement.text=str(os.path.join("",filenmeelement.text))
            print "\n{}.jpg".format(str(os.path.join("/home/wang/anpr/data",filenameelement.text)))

if __name__ == "__main__":
    walkdir(sys.argv[1])


