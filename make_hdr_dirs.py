''' requirements for this code**
pip install opencv-python
pip install numpy
pip install OpenEXR
pip install Imath
'''
import os
import cv2
import numpy as np
import shutil
import rawpy
import imageio
import datetime

dataroot = './pre_hdr/3_31_24_fog'
output_dataroot = './post_hdr/hdr_images/fog'


def make_dir_hdr(dataroot):
    ''' this function deals with .raw to .bmp conversion and
    makes individual directories for each set of 4 images to be processed
    HDR this is a helper script for ordinizational processes it into also
    helps with debugging and keeping track of the images as you can easily
    see which images are in which set and ajust offsets accordingly.'''
    __temp = (len(os.listdir(dataroot)))//4
    for i in range(__temp):
        os.makedirs(f'{dataroot}/{i}' , exist_ok=True)

    for file in os.listdir(dataroot): 
        # if files are in raw format they are converted to bmp
        if file.endswith('.raw'): 
            raw = rawpy.imread(f'{dataroot}/{file}')
            temp = file.strip('.raw')
            temp = int(temp)
            temp = (temp-1)//4
            rgb = raw.postprocess()
            new_path = f'{dataroot}/{temp}/{file.strip('.raw')}.bmp'
            imageio.imsave(new_path, rgb)
            
        if file.endswith('.bmp'): # for if files are already in bmp format
            temp = file.strip('.bmp')
            temp = int(temp)
            temp = (temp-1)//4
            shutil.move(f'{dataroot}/{file}', f'{dataroot}/{temp}/{file}') 




print(datetime.date.today())

# Path to the folder containing the images to be processed into HDR
#dataroot = "./pre_hdr/3_31_24_fog"
#out_dataroot = "./post_hdr/hdr_images"

def create_hdr_images(dataroot, out_dataroot='./post_hdr/hdr_images', HDR_type='Debevec'):
    ''' This function creates HDR images from the images in the dataroot folder
    '''
    exposure = np.array([0.5, 0.75, 1.5, 2], dtype=np.float32)
    _temp = []
    for file in os.listdir(dataroot):
        if file.endswith('.txt'):
            _temp.append(file)
    if not len(_temp) == 1:
        raise AssertionError("\nThere should be only one .txt file in the dataroot folder.\nThis .txt file should contain the exposure times for the images in the folder.\nif you dont have the exposure times consider changing the HDR_type to 'Mertens' or 'Robertson'.\n")

    with open(f'{dataroot}/{_temp[0]}', 'r') as f:
        exposure_ts = f.readlines()


    if not os.path.exists(out_dataroot):
        os.makedirs(out_dataroot)
    # Assuming each image set to be processed into HDR is in a separate subfolder
    for subdir in os.listdir(dataroot):
        subdir_path = os.path.join(dataroot, subdir)
        if os.path.isdir(subdir_path):
            #print(subdir_path.split('\\')[-1])
            temp_idx = int(subdir_path.split('\\')[-1])
            #assert True == False
            images = []
            for filename in sorted(os.listdir(subdir_path), key=lambda x: int(x.split('.')[0])):
                file_path = os.path.join(subdir_path, filename)
                im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)  # Adjust if your images are not standard 8-bit or 16-bit images
                if im is not None:
                    images.append(im)

            if len(images) > 0:
                # Align input images
                alignMTB = cv2.createAlignMTB()
                alignMTB.process(images, images)
                # Obtain Camera Response Function (CRF)
                exposure_times = exposure*float(exposure_ts[temp_idx])*0.000001

                # 1 - MergeDebevec
                if HDR_type == 'Debevec':
                    if not os.path.exists(f"{out_dataroot}/vga"):
                        os.makedirs(f"{out_dataroot}/vga")
                        os.makedirs(f"{out_dataroot}/vga/22")
                        os.makedirs(f"{out_dataroot}/vga/44")
                        os.makedirs(f"{out_dataroot}/vga/10")

                    # Note: HDR images have a high dynamic range that cannot be properly displayed on standard monitors
                    # without tone mapping. Here, we'll just visualize a tonemapped version for simplicity.
                    calibrateDebevec = cv2.createCalibrateDebevec()
                    responseDebevec = calibrateDebevec.process(images, exposure_times)
                    mergeDebevec = cv2.createMergeDebevec()
                    hdrDebevec = mergeDebevec.process(images, exposure_times, responseDebevec)
                    hdr_filename = os.path.join(out_dataroot, f"vga/vga_hdr_{datetime.date.today()}_{subdir}.hdr")
                    cv2.imwrite(hdr_filename, hdrDebevec.copy())
                    # Save your HDR tonemapped images at 3 different gamma values 
                    hdr_filename = os.path.join(out_dataroot, f"vga/22/vga_hdr_{datetime.date.today()}_{subdir}_22.bmp")
                    tonemapped = cv2.createTonemap(2.2).process(hdrDebevec.copy())
                    tonemapped = np.clip(tonemapped*255, 0, 255).astype('uint8')
                    cv2.imwrite(hdr_filename, tonemapped)
                    if subdir in '1':
                        cv2.imshow('HDR Image', tonemapped)
                        cv2.waitKey()

                    tonemapped = cv2.createTonemap(4.4).process(hdrDebevec.copy())
                    tonemapped = np.clip(tonemapped*255, 0, 255).astype('uint8')
                    hdr_filename = os.path.join(f"{out_dataroot}", f"vga/44/vga_hdr_{datetime.date.today()}_{subdir}_44.bmp")
                    cv2.imwrite(hdr_filename, tonemapped)
                    tonemapped = cv2.createTonemap(1.0).process(hdrDebevec.copy())
                    tonemapped = np.clip(tonemapped*255, 0, 255).astype('uint8')
                    hdr_filename = os.path.join(out_dataroot, f"vga/10/vga_hdr_{datetime.date.today()}_{subdir}_10.bmp")
                    cv2.imwrite(hdr_filename, tonemapped)


                # 2 - MergeRobertson has not been tested    ###################
                if HDR_type == 'Robertson':
                    mergeRobertson = cv2.createMergeRobertson()
                    hdrRobertson = mergeRobertson.process(images, exposure_times)
                    hdr_filename = os.path.join(out_dataroot, f"hdr_{datetime.date.strftime("%Y-%m-%d")}_{subdir}.bmp")
                    hdrRobertson = cv2.createTonemap(2.2).process(hdrRobertson.copy())
                    #hdrRobertson = cv2.normalize(hdrRobertson, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    cv2.imwrite(hdr_filename, hdrRobertson.copy())


                # 3 - MergeMertens Does not require exposure times
                if HDR_type == 'Mertens':
                    mergeMertens = cv2.createMergeMertens()
                    hdrMertens = mergeMertens.process(images)
                    res_16bit = np.clip(hdrMertens*255, 0, 255).astype('uint8')
                    hdr_filename = os.path.join(out_dataroot, f"hdr_{datetime.date.strftime("%Y-%m-%d")}_{subdir}.bmp")
                    cv2.imwrite(hdr_filename, res_16bit)
                    # Display the HDR image

            print(f"Saved HDR image to {hdr_filename}")
        cv2.destroyAllWindows()



if __name__ == '__main__':
    make_dir_hdr(dataroot)
    create_hdr_images(dataroot, out_dataroot=output_dataroot)

