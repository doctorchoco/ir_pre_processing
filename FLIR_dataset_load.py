import json


dirs = "D:/IR_data/FLIR_ADAS_1_3/train/thermal_annotations.json"



with open(dirs, "r") as st_json:
    label = json.load(st_json)

vDict = label

annotation = vDict["annotations"]
images = vDict["images"]

# info
# categories
# licenses
# annotations
# images

for i in range(len(images)):
    print(images[i])

#    print("Key:%s\tValue:%d" % (item, vDict[item]))