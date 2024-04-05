''' requirements for this code**
pip install opencv-python
pip install numpy
pip install OpenEXR
pip install Imath
'''
import os
import cv2
import numpy as np
import Imath
import OpenEXR


def save_exr(path, image):
    """
    Save a 32-bit floating point image as an EXR file.
    Args:
        path: File path to save the EXR file.
        image: A numpy array representing the HDR image.
    """
    height, width = image.shape[:2]
    HEADER = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])
    out = OpenEXR.OutputFile(path, HEADER)
    R = (image[:,:,0]).astype(np.float32).tobytes()
    G = (image[:,:,1]).astype(np.float32).tobytes()
    B = (image[:,:,2]).astype(np.float32).tobytes()
    out.writePixels({'R': R, 'G': G, 'B': B})
    out.close()
# Save your HDR image as an EXR file

# Path to the folder containing the images to be processed into HDR
data_folder = "./pre_hdr/3_13_24_fog"
output_folder = "./post_hdr/hdr_images"
exposure = np.array([0.5, 0.75, 1.5, 2], dtype=np.float32)
with open(data_folder + '/times_3_13.txt', 'r') as f:
    exposure_ts = f.readlines()

#exposure_times = np.array([exposure[0], exposure[1], exposure[2]], dtype=np.float32)  # Example exposure times
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Assuming each image set to be processed into HDR is in a separate subfolder
for subdir in os.listdir(data_folder):
    subdir_path = os.path.join(data_folder, subdir)
    if os.path.isdir(subdir_path):
        print(subdir_path.split('\\')[-1])
        temp_idx = int(subdir_path.split('\\')[-1])
        #assert True == False
        images = []
        for filename in sorted(os.listdir(subdir_path), key=lambda x: int(x.split('.')[0])):
            file_path = os.path.join(subdir_path, filename)
            im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)  # Adjust if your images are not standard 8-bit or 16-bit images
            if im is not None:
                images.append(im)
                print(filename)
        if len(images) > 0:
            # Align input images
            alignMTB = cv2.createAlignMTB()
            alignMTB.process(images, images)
            # Obtain Camera Response Function (CRF)
            calibrateDebevec = cv2.createCalibrateDebevec()
            exposure_times = exposure*float(exposure_ts[temp_idx])*0.000001
            #print(exposure_times)
            responseDebevec = calibrateDebevec.process(images, exposure_times)
            # Merge images into an HDR linear image there are 3 options for this
            # 1 - MergeDebevec
            mergeDebevec = cv2.createMergeDebevec()
            hdrDebevec = mergeDebevec.process(images, exposure_times, responseDebevec)
            # 2 - MergeRobertson
            # mergeRobertson = cv2.createMergeRobertson()
            # hdrRobertson = mergeRobertson.process(images, exposure_times)
            # 3 - MergeMertens
            #mergeMertens = cv2.createMergeMertens()
            #hdrMertens = mergeMertens.process(images)
            # Display the HDR image
            # Note: HDR images have a high dynamic range that cannot be properly displayed on standard monitors
            # without tone mapping. Here, we'll just visualize a tonemapped version for simplicity.
            hdr_filename = os.path.join(output_folder, f"vga/vga_hdr_{subdir}.hdr")
            cv2.imwrite(hdr_filename, hdrDebevec.copy())

            hdr_filename = os.path.join(output_folder, f"vga/22/vga_hdr_{subdir}_22.bmp")
            tonemapped = cv2.createTonemap(2.2).process(hdrDebevec.copy())
            tonemapped = np.clip(tonemapped*255, 0, 255).astype('uint8')
            cv2.imwrite(hdr_filename, tonemapped)
            if subdir in '01234':
                cv2.imshow('HDR Image', tonemapped)
                cv2.waitKey()
            
            tonemapped = cv2.createTonemap(4.4).process(hdrDebevec.copy())
            tonemapped = np.clip(tonemapped*255, 0, 255).astype('uint8')
            hdr_filename = os.path.join(f"{output_folder}", f"vga/44/vga_hdr_{subdir}_44.bmp")
            cv2.imwrite(hdr_filename, tonemapped)
            #cv2.imshow('HDR Image', tonemapped)
            #cv2.waitKey()
            tonemapped = cv2.createTonemap(1.0).process(hdrDebevec.copy())
            tonemapped = np.clip(tonemapped*255, 0, 255).astype('uint8')
            hdr_filename = os.path.join(output_folder, f"vga/10/vga_hdr_{subdir}_10.bmp")
            cv2.imwrite(hdr_filename, tonemapped)
            #cv2.imshow('HDR Image', tonemapped)
            #cv2.waitKey()
            # Save HDR image
            #res_16bit = np.clip(hdrMertens*255, 0, 255).astype('uint16')
            #hdr_filename = os.path.join(output_folder, f"hdr_{subdir}.hdr")
            #cv2.imwrite(hdr_filename, hdrDebevec)
            #
            #save_exr(hdr_filename, hdrDebevec)
            print(f"Saved HDR image to {hdr_filename}")
cv2.destroyAllWindows()

#save_exr(output_folder+'\\'+'output_image.png', hdrDebevec)