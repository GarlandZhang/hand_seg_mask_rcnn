import shutil
import os

# dir
direc = '../egohands_kitti_formatted/labels'

file_names = os.listdir(direc)
file_names.sort()

for i, f in enumerate(file_names):
  ind = str(i + 1)
  num_zeros = 5 - len(ind)
  for j in range(0, num_zeros):
    ind = "0" + ind

  file_ext = os.path.splitext(f)[-1]
  new_file_name = ind + file_ext
  shutil.move(os.path.join(direc, f), os.path.join(direc, new_file_name))