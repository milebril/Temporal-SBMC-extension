import numpy as np
from PIL import Image

image_low = Image.open('/home/emil/Documents/Temporal-SBMC-extension/output/emil/test_set/samples/cornell_paper/0000_low_spp.png')
image_up = Image.open('/home/emil/Documents/Temporal-SBMC-extension/output/emil/test_set/samples/cornell_paper/-peters-scene.png')

image_low = np.asarray(image_low)
image_up = np.asarray(image_up)

image_low_part = np.zeros_like(image_low)
image_up_part = np.zeros_like(image_up)

for i in range(image_low.shape[2]):
    image_low_part[:,:,i] = np.tril(image_low[:,:,i], k=-2)

for i in range(image_up.shape[2]):
    image_up_part[:,:,i] = np.triu(image_up[:,:,i], k=2)

# final = Image.fromarray(np.uint8(image_low_part))

final = image_low_part + image_up_part
final = Image.fromarray(np.uint8(final))

final.save('/home/emil/Documents/Temporal-SBMC-extension/output/emil/test_set/samples/cornell_paper/final.png')