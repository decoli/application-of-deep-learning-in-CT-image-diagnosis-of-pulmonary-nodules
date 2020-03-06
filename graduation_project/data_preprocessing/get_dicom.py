import pydicom
import cv2

path_dicom = 'data/lidc/image/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192/000094.dcm'
d = pydicom.read_file(path_dicom)
print(d)

image_array = d.pixel_array

###
image_array[image_array > 600] = 0
image_array[image_array < -400] = 0

###
# max = image_array.max()
# min = image_array.min()
# image_array = (image_array-min)/(max-min) 
# avg = image_array.mean()
# image_array = image_array-avg
# image_array = image_array * 255

###
# image_array = (image_array - (-1200)) / (600 - (-1200))
# image_array = image_array * 255

###
path_image = 'test_dicom.png'
cv2.imwrite(path_image, image_array)

print('ttt')