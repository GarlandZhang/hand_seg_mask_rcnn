import xml.etree.ElementTree as ET
import os
import cv2

direc = 'E://egohands_kitti_formatted/images'
labels_direc = 'E://egohands_kitti_formatted/annots'

for f in os.listdir(direc):
  img = cv2.imread(os.path.join(direc, f))
  h, w, c = img.shape

  base_name, file_ext = os.path.splitext(f)

  annotation = ET.Element('annotation')
  folder = ET.SubElement(annotation, 'folder')
  folder.text = 'hands'
  filename = ET.SubElement(annotation, 'filename')
  
  filename.text = f
  path = ET.SubElement(annotation, 'path')
  path.text = 'who/cares'
  source = ET.SubElement(annotation, 'source')
  database = ET.SubElement(source, 'database')
  database.text = 'Unknown'
  size = ET.SubElement(annotation, 'size')
  width = ET.SubElement(size, 'width')
  width.text = str(w)
  height = ET.SubElement(size, 'height')
  height.text = str(h)
  depth = ET.SubElement(size, 'depth')
  depth.text = str(c)
  segmented = ET.SubElement(annotation, 'segmented')
  segmented.text = "0"

  # get box info from text file
  boxes = []
  with open(os.path.join(labels_direc, base_name + '.txt'), 'r+') as file_reader:
    while True:
      line = file_reader.readline()
      if not line:
        break
      vals = [int(val) for val in line.split(' ')[1:]]
      
      if len(vals) != 0:
        box = vals[3:7]
        boxes.append(box)

  for xmin, ymin, xmax, ymax in boxes:
    obj = ET.SubElement(annotation, 'object')
    name = ET.SubElement(obj, 'name')
    name.text = 'hand'
    bndbox = ET.SubElement(obj, 'bndbox')
    xm = ET.SubElement(bndbox, 'xmin')
    xm.text = str(xmin)
    ym = ET.SubElement(bndbox, 'ymin')
    ym.text = str(ymin)
    xma = ET.SubElement(bndbox, 'xmax')
    xma.text = str(xmax)
    yma = ET.SubElement(bndbox, 'ymax')
    yma.text = str(ymax)

  file_path = os.path.join(labels_direc, base_name + '.xml')
  
  with open(file_path, 'wb') as file_writer:
    data = ET.tostring(annotation)
    file_writer.write(data)