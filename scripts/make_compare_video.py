import cv2
import numpy as np
import glob

noisy = []
gharbi = []
peters = []

final = []

for filename in sorted(glob.glob('/home/emil/Documents/Temporal-SBMC-extension/output/emil/dataviz_sequence/*.png')):
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    noisy.append(img)

for filename in sorted(glob.glob('/home/emil/Documents/Temporal-SBMC-extension/output/emil/dataviz_sequence/denoised/pretrained/*.png')):
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    gharbi.append(img)

for filename in sorted(glob.glob('/home/emil/Documents/Temporal-SBMC-extension/output/emil/dataviz_sequence/denoised/peters/*.png')):
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    peters.append(img)

for i in range(len(noisy)):
    final.append(np.concatenate((noisy[i], gharbi[i], peters[i]), axis=0))

height, width, layers = final[0].shape
# Write to video
out = cv2.VideoWriter('/home/emil/Documents/Temporal-SBMC-extension/output/emil/dataviz_sequence/video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, (width, height))
 
# Stitch 5 times to create loop
for _ in range(5):
    for i in range(len(final)):
        out.write(final[i])

out.release()