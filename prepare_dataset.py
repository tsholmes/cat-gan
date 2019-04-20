import glob
import os
import subprocess
from PIL import Image

if not os.path.exists("CAT_00"):
    print("Downloading cat dataset...")
    subprocess.check_output("wget https://archive.org/download/CAT_DATASET/CAT_DATASET_01.zip", shell=True)
    subprocess.check_output("unzip CAT_DATASET_01.zip", shell=True)
    os.mkdir("cat_crop")

count = 0
bounds_filenames = glob.glob("CAT_*/*.jpg.cat")
for filename in bounds_filenames:
    with open(filename, 'r') as f:
        contents = f.read()
        nums = [int(n) for n in contents.split(' ') if len(n) > 0]
    if nums[0] != 9:
        continue
    nums = nums[1:]
    min_x, max_x, min_y, max_y = 100000, -1, 100000, -1
    for i in range(len(nums)):
        if i % 2 == 0:
            min_x = min(min_x, nums[i])
            max_x = max(max_x, nums[i])
        else:
            min_y = min(min_y, nums[i])
            max_y = max(max_y, nums[i])
    if max_x - min_x < 100 or max_y - min_y < 100:
        continue
    
    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2
    
    sz = max(128, int((max_x - min_x) * 0.6), int((max_y - min_y) * 0.6))
    
    if cx < sz or cy < sz:
        continue
    
    img_filename = filename[:-4]
    # bug in dataset
    if img_filename == 'CAT_00/00000003_019.jpg':
        img_filename = 'CAT_00/00000003_015.jpg'
    
    new_filename = 'cat_crop/' + img_filename[7:]
    
    img = Image.open(img_filename)
    
    if cx + sz > img.width or cy + sz > img.height:
        continue
    
    print('%s => %s (%d)' % (img_filename, new_filename, sz))
    img.crop((cx - sz, cy - sz, cx + sz, cy + sz)).resize((256, 256)).save(new_filename)
    count = count + 1

print(count)