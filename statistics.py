# **change path of your xml folder
# **this program can help you do the statistic summury of 
#      how many annotations and images for each catogorys
#      in your dataset 


import csv
import os
import  xml.dom.minidom
from copy import deepcopy


def get_xmlnode(node, name):
    return node.getElementsByTagName(name) if node else []

def get_nodevalue(node, index = 0):
    return node.childNodes[index].nodeValue if node else ''

path = "./ano0_train000_xml"
result = os.walk(path)
name_set = set()
# name_num = {"person":0, "car":0, "cone": 0, "umbrellaman":0, "bicycle":0,"dog":0}
# name_num = {'background':0,
#            'aeroplane':0, 'bicycle':0, 'bird':0, 'boat':0,
#            'bottle':0, 'bus':0, 'car':0, 'cat':0,
#            'chair':0, 'cow':0, 'diningtable':0, 'dog':0,
#            'horse':0, 'motorbike':0, 'person':0, 'pottedplant':0,
#            'sheep':0, 'sofa':0, 'train':0, 'tvmonitor':0}
name_num = {'aeroplane':0, 'bicycle':0 ,'bird':0 ,'boat':0,
          'bus':0,'car':0,'cat':0 ,'cow':0 ,'dog':0,'horse':0,
          'motorbike':0 ,'person':0 ,'sheep':0 ,'train':0 ,
          'umbrellaman':0 ,'cone':0 }
img_name_num =deepcopy(name_num)
calculate = 0     
for a in result:
    for file_x in a[2]:
        file_path = os.path.join(path, file_x)
        dom = xml.dom.minidom.parse(file_path)
        root = dom.documentElement
        filename = root.getElementsByTagName("filename")
        filename = get_nodevalue(filename[0])
        node = get_xmlnode(root, "object")
        list_tem = []

        name_num_before = deepcopy(name_num)

        for node_tem in node:
            name = get_xmlnode(node_tem, "name")
            name = get_nodevalue(name[0])

            name_set.add(name)
            name_num[name] = name_num[name] + 1

        for name in name_num:
            if name_num[name] > name_num_before[name]:
                img_name_num[name] = img_name_num[name] + 1
        
        calculate = calculate +1

        print(file_x, "---------is done!",calculate)
        # print(img_name_num)

print('------------------train000(my)-finish----------------------')
print('------------------object_num----------------------')
print(name_num)
print('---------------img_name_num-------------------')
print(img_name_num)



