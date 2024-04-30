''' requirements for this code**
pip install opencv-python
pip install numpy
pip install OpenEXR
pip install Imath
'''
import argparse
import os
import cv2
import numpy as np
import shutil
import rawpy
import imageio
import datetime
import subprocess

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
            params = rawpy.Params(output_bps=16, no_auto_bright=True, use_camera_wb=True, use_auto_wb=False, gamma=(1,1), no_auto_scale=True, user_flip=0)
            #raw.color_desc = 'BGGR'
            temp = file.strip('.raw')
            temp = int(temp)
            temp = (temp-1)//4
            rgb = raw.postprocess()
            new_path = f'{dataroot}/{temp}/{file.strip(".raw")}.bmp'
            #bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = rgb # if we are useing the same module no need to switch.
            cv2.imwrite(new_path, bgr)
            #imageio.imsave(new_path, bgr)
            
        if file.endswith('.bmp'): # for if files are already in bmp format
            temp = file.strip('.bmp')
            temp = int(temp)
            temp = (temp-1)//4
            shutil.move(f'{dataroot}/{file}', f'{dataroot}/{temp}/{file}') 




#print(datetime.date.today())

# Path to the folder containing the images to be processed into HDR
#dataroot = "./pre_hdr/3_31_24_fog"
#out_dataroot = "./post_hdr/hdr_images"
#import multiprocessing
############################# should really add multiprocessing to this function. ##################################
############## for when the dataset gets larger so you are not running everything on one core. #####################
#multiprocessing.SimpleQueue()
"""def create_camera_response_function(dataroot='./pre_hdr'):
    ''' This function creates the camera response function for the images in the dataroot folder
    it only uses the truth images to create the camera response function.
    as using foggy images would not be ideal for this process and can create a bad camera response function.'''
    all_response = np.array([])
    for child in os.listdir(dataroot):
        #if child.endswith('truth'):
        if child == '3_13_24_truth':
            dataroot_temp = os.path.join(dataroot, child)
            #make_dir_hdr(dataroot_temp)
            child = f'truth\\{child.strip("truth")}'
            is_fog = False

        else:
            continue
        
        
        exposure = np.array([0.5, 0.75, 1.5, 2], dtype=np.float32)
        _temp = []
        for file in os.listdir(dataroot_temp):
            if file.endswith('.txt'):
                _temp.append(file)
        if not len(_temp) == 1:
            raise AssertionError("\nThere should be only one .txt file in the dataroot_temp folder.\nThis .txt file should contain the exposure times for the images in the folder.\nif you dont have the exposure times consider changing the HDR_type to 'Mertens' or 'Robertson'.\n")

        with open(f'{dataroot_temp}/{_temp[0]}', 'r') as f:
            exposure_ts = f.readlines()
        
        # Assuming each image set to be processed into HDR is in a separate subfolder
        for subdir in os.listdir(dataroot_temp):
            subdir_path = os.path.join(dataroot_temp, subdir)
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
                    alignMTB = cv2.createAlignMTB()
                    alignMTB.process(images, images)
                    exposure_times = exposure*float(exposure_ts[temp_idx])*0.000001
                    calibrateDebevec = cv2.createCalibrateDebevec()
                    responseDebevec = calibrateDebevec.process(images, exposure_times)
                    print(responseDebevec)
                    all_response.append(responseDebevec)
    all_response = np.array(all_response)# shape=(len(all_response), 256, 3))
    calibrated_responce = all_response.mean(axis=0)
    if responseDebevec.shape == calibrated_responce.shape:
        print("Camera response function created successfully.")
    calibrated_responce(calibrated_responce)  """
            


def create_hdr_images(dataroot='./pre_hdr', out_dataroot='./post_hdr/hdr_images', HDR_type='Debevec', is_fog=False):
    ''' This function creates HDR images from the images in the dataroot folder
    '''
    for child in os.listdir(dataroot):
        """if child.endswith('fog'):
            dataroot_temp = os.path.join(dataroot, child)
            #make_dir_hdr(dataroot_temp)
            child = f'fog\\{child.strip("fog")}'
            is_fog = True"""

        if child.endswith('truth'):
            dataroot_temp = os.path.join(dataroot, child)
            fog_dataroot_temp = os.path.join(dataroot, child.strip('truth')+'fog')
            make_dir_hdr(dataroot_temp)
            make_dir_hdr(fog_dataroot_temp)
            child = f'truth\\{child.strip("truth")}'
            _child_fog = f'fog\\{child.strip("truth")}'
            is_fog = False

        else:
            continue
        
        
        exposure = np.array([0.5, 0.75, 1.5, 2], dtype=np.float32)
        _temp = []
        _temp_fog = []
        for file in os.listdir(dataroot_temp):
            if file.endswith('.txt'):
                _temp.append(file)
        for _file in os.listdir(dataroot_temp.split('truth')[0]+'fog'):
            if _file.endswith('.txt'):
                _temp_fog.append(file)
        print(_temp, _temp_fog)
        if not len(_temp) == 1:
            raise AssertionError("\nThere should be only one .txt file in the dataroot_temp folder.\nThis .txt file should contain the exposure times for the images in the folder.\nif you dont have the exposure times consider changing the HDR_type to 'Mertens' or 'Robertson'.\n")

        with open(f'{dataroot_temp}/{_temp[0]}', 'r') as f:
            exposure_ts = f.readlines()
        with open(f'{fog_dataroot_temp}/{_temp_fog[0]}', 'r') as f:    
            _exposure_ts = f.readlines()

        #print(out_dataroot)
        if not os.path.exists(out_dataroot):
            os.makedirs(out_dataroot)
        
        # Assuming each image set to be processed into HDR is in a separate subfolder
        responces= []
        for subdir in os.listdir(dataroot_temp):
            subdir_path = os.path.join(dataroot_temp, subdir)
            subdir_path_fog = os.path.join(fog_dataroot_temp, subdir)
            if os.path.isdir(subdir_path):
                #print(subdir_path.split('\\')[-1])
                temp_idx = int(subdir_path.split('\\')[-1])
                #assert True == False
                images = []
                fog_images = []
                for filename in sorted(os.listdir(subdir_path), key=lambda x: int(x.split('.')[0])):
                    file_path = os.path.join(subdir_path, filename)
                    im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)  # Adjust if your images are not standard 8-bit or 16-bit images
                    im_fog = cv2.imread(os.path.join(subdir_path_fog, filename), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                    if im is not None:
                        images.append(im)
                    if im_fog is not None:
                        fog_images.append(im_fog)
                """if os.path.isdir(subdir_path_fog):
                
                for filename in sorted(os.listdir(subdir_path), key=lambda x: int(x.split('.')[0])):
                    file_path = os.path.join(subdir_path, filename)
                    im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)  # Adjust if your images are not standard 8-bit or 16-bit images
                    if im is not None:
                        images.append(im)"""




                if len(images) > 0 and len(fog_images) > 0:

                    ################## Align images ##################
                    # Note: If you are using a tripod, you can skip this step
                    # it was not working for the foggy images and caused us much heartache
                    # as we did not notice until the end of the project. 04-25-2024
                    exposure_times = exposure*float(exposure_ts[temp_idx])*0.000001
                    _exposure_times = exposure*float(_exposure_ts[temp_idx])*0.000001
                    if not is_fog:
                        alignMTB = cv2.createAlignMTB()
                        """shift the fogged images and the truth images to be aligned with each other."""
                        """for i in range(len(images)):
                            alignMTB.calculateShift(images[i-1], images[i])
                            alignMTB.shift(fog_images[i-1], fog_images[i])"""
                        
                        
                        ''' translation = alignMTB.calculateShift(base_images[0], images[0])
                        alignMTB.shiftMat(fog_images,translation, fog_images)
                        cv2.imshow('image', fog_images[0])
                        cv2.waitKey()
                        alignMTB.shiftMat(images[0],translation, images[0])

                        translation = alignMTB.calculateShift(images[2], images[1])
                        alignMTB.shiftMat(images[1],translation, images[1])
                        alignMTB.shiftMat(fog_images[1],translation, fog_images[1])

                        translation = alignMTB.calculateShift(images[2], images[3])
                        alignMTB.shiftMat(images[3],translation, images[3])
                        alignMTB.shiftMat(fog_images[3],translation, fog_images[3])'''




                        alignMTB.process(images, images)
                        
                        
                    #alignMTB = cv2.createAlignMTB()
                    #alignMTB.process(images, images)

    ################## Obtain Camera Response Function (CRF) ##################
    # this is saved for future if the proof of concept works.
    # you can also use the code that is commented out above to make aligned images 
    # if you are no longer using a tripod. I am starting to get the average inverse camera responce
    # and saving that to a file in hopes of future use. I am not sure if this will be usable or if it 
    # is a waste, but further analysis should help decide. also Alighment of foggy images is still an issue

                    
                    # 1 - MergeDebevec
                    if HDR_type == 'Debevec':
                        if not os.path.exists(f"{out_dataroot}\\22\\fog"):
                            os.makedirs(f"{out_dataroot}\\22\\fog")
                            os.makedirs(f"{out_dataroot}\\22\\truth")
                            os.makedirs(f"{out_dataroot}\\44\\fog")
                            os.makedirs(f"{out_dataroot}\\44\\truth")
                            os.makedirs(f"{out_dataroot}\\10\\fog")
                            os.makedirs(f"{out_dataroot}\\10\\truth")

                        # Note: HDR images have a high dynamic range that cannot be properly displayed on standard monitors
                        # without tone mapping. Here, we'll just visualize a tonemapped version for simplicity.

                        ################################## calibrateDebevec fog makes the color weird   ##################################
                        #trying to find the inverse camera response function for the fogged images is what is making them weird colors.
                        calibrateDebevec = cv2.createCalibrateDebevec()
                        responseDebevec = calibrateDebevec.process(images, exposure_times)
                        mergeDebevec = cv2.createMergeDebevec()
                        hdrDebevec = mergeDebevec.process(images, exposure_times, responseDebevec.copy())
                        _fog_hdrDebevec = mergeDebevec.process(fog_images, _exposure_times, responseDebevec.copy())
                        #hdr_filename = os.path.join(out_dataroot, f"\\22\\{child}{subdir}.hdr")
                        #cv2.imwrite(hdr_filename, hdrDebevec.copy())
                        # Save your HDR tonemapped images at 3 different gamma values 
                        hdr_filename = os.path.join(out_dataroot, f"22\\{child}){subdir}_22.bmp")
                        tonemapped = cv2.createTonemap(2.2).process(hdrDebevec.copy())
                        tonemapped = np.clip(tonemapped*255, 0, 255).astype('uint8')
                        cv2.imwrite(hdr_filename, tonemapped)
                        #alignMTB= cv2.createAlignMTB()
                        #alignMTB.process(fog_images, fog_images, times=_exposure_times, response=responseDebevec.copy())
                        mergeDebevec = cv2.createMergeDebevec()
                        _fog_hdrDebevec = mergeDebevec.process(fog_images, _exposure_times, responseDebevec.copy())

                        fog_hdr_filename = os.path.join(out_dataroot, f"22\\{_child_fog}{subdir}_22.bmp")
                        fog_tonemapped = cv2.createTonemap(2.2).process(_fog_hdrDebevec.copy())
                        fog_tonemapped = np.clip(fog_tonemapped*255, 0, 255).astype('uint8')
                        cv2.imwrite(fog_hdr_filename, fog_tonemapped)
                        
                        if False:
                            cv2.imshow('HDR Image', fog_tonemapped)
                            cv2.waitKey()
                        responces.append(responseDebevec)

                        ############### fix once I have the fogged images aligned with the truth images. ################
                        """tonemapped = cv2.createTonemap(4.4).process(hdrDebevec.copy())
                        tonemapped = np.clip(tonemapped*255, 0, 255).astype('uint8')
                        hdr_filename = os.path.join(out_dataroot, f"44\\{child}{subdir}_44.bmp")
                        cv2.imwrite(hdr_filename, tonemapped)
                        tonemapped = cv2.createTonemap(1.0).process(hdrDebevec.copy())
                        tonemapped = np.clip(tonemapped*255, 0, 255).astype('uint8')
                        hdr_filename = os.path.join(out_dataroot, f"10\\{child}{subdir}_10.bmp")
                        cv2.imwrite(hdr_filename, tonemapped)"""

                    # 2 - MergeRobertson has not been tested    ###################
                    if HDR_type == 'Robertson':
                        mergeRobertson = cv2.createMergeRobertson()
                        hdrRobertson = mergeRobertson.process(images, exposure_times)
                        hdr_filename = os.path.join(out_dataroot, f"hdr_{dataroot_temp}{subdir}.bmp")
                        hdrRobertson = cv2.createTonemap(2.2).process(hdrRobertson.copy())
                        #hdrRobertson = cv2.normalize(hdrRobertson, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        cv2.imwrite(hdr_filename, hdrRobertson.copy())


                    # 3 - MergeMertens Does not require exposure times
                    if HDR_type == 'Mertens':
                        mergeMertens = cv2.createMergeMertens()
                        hdrMertens = mergeMertens.process(images)
                        res_16bit = np.clip(hdrMertens*255, 0, 255).astype('uint8')
                        hdr_filename = os.path.join(out_dataroot, f"hdr_{dataroot_temp}{subdir}.bmp")
                        cv2.imwrite(hdr_filename, res_16bit)
                        # Display the HDR image

                #print(f"Saved HDR image to {hdr_filename}")
            cv2.destroyAllWindows()
    with open(f'{out_dataroot}\\response.txt', 'w') as f:
        for responce in responces:
            f.write(f'{responce}\n')



######### These Main functions need to be edited to that they act the way that we want them to. 
# the goal for this file is for it to be a utility function file.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='The root directory of the dataset')
    #parser.add_argument('--out_dataroot', type=str, required=True, help='The path to the images without fog')
    #parser.add_argument('--HDR_type', type=str, default='Debevec', help='The type of HDR to create')
    #parser.add_argument('--is_fog', type=bool, default=False, help='Whether the images are foggy or not')
    args = parser.parse_args()

    create_hdr_images(args.dataroot)
    #create_camera_response_function(args.dataroot)
    #if not os.path.exists(args.out_dataroot):
    #    os.makedirs(args.out_dataroot)
    #is_fog = args.is_fog
    #create_hdr_images(args.dataroot, out_dataroot=args.out_dataroot, HDR_type='Debevec', is_fog=is_fog)

######### These Main functions need to be edited to that they act the way that we want them to. 
# the goal for this file is for it to be a utility function file.

'''if __name__ == '__main__':
    file_path_1 = './datasets/stereofog_images/final_project_dataset/B'
    file_path_2 = './datasets/stereofog_images/final_project_dataset/A'
    new_dir = './datasets/stereofog_images/final_project_dataset/no_match2'
    #dataroot = '.\\post_hdr\\final\\A'
    #identifier = '04-20'
    #rename_matching_files(dataroot, split_identifier=identifier)
    print(check_match(file_path_1, file_path_2, new_dir))'''



"""


''' requirements for this code**
pip install opencv-python
pip install numpy
pip install OpenEXR
pip install Imath
'''
import argparse
import os
import cv2
import numpy as np
import shutil
import rawpy
import imageio
import datetime
import subprocess


# this script will look in file A and file B and any files
# that are in A but not B will be moved to a new directory
def check_match(file_path_1, file_path_2, new_dir):
    # get the list of files in the first directory
    file_list_1 = os.listdir(file_path_1)
    # get the list of files in the second directory
    file_list_2 = os.listdir(file_path_2)
    # create a list of files that are in the first directory but not the second
    not_in_2 = [file for file in file_list_1 if file not in file_list_2]
    # create a new directory to move the files to
    os.mkdir(new_dir)
    # move the files from the first directory to the new directory
    for file in not_in_2:
        os.rename(file_path_1 + '/' + file, new_dir + '/' + file)
    return not_in_2

def rename_matching_files(dataroot, split_identifier='_', new_dir='Combined'):
    temp = 0
    os.makedirs(f'{dataroot}/{new_dir}', exist_ok=True)
    for subdir in os.listdir(dataroot):
        print(subdir)
        temp +=1
        if subdir == new_dir:
            continue
        for temp_file in os.listdir(f'{dataroot}/{subdir}'):
            
            if temp_file.endswith('.bmp'):
                t = temp_file.split(split_identifier)[-1]
                new_file = f'{temp}{t}'
                shutil.copy(f'{dataroot}/{subdir}/{temp_file}', f'{dataroot}/{new_dir}/{new_file}')


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
            params = rawpy.Params(output_bps=16, no_auto_bright=True, use_camera_wb=True, use_auto_wb=False, gamma=(1,1), no_auto_scale=True, user_flip=0)
            #raw.color_desc = 'BGGR'
            temp = file.strip('.raw')
            temp = int(temp)
            temp = (temp-1)//4
            rgb = raw.postprocess()
            new_path = f'{dataroot}/{temp}/{file.strip('.raw')}.bmp'
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            imageio.imsave(new_path, bgr)
            
        if file.endswith('.bmp'): # for if files are already in bmp format
            temp = file.strip('.bmp')
            temp = int(temp)
            temp = (temp-1)//4
            shutil.move(f'{dataroot}/{file}', f'{dataroot}/{temp}/{file}') 




#print(datetime.date.today())

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
                    if subdir in '':
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
                    hdr_filename = os.path.join(out_dataroot, f"hdr_{datetime.date.today()}_{subdir}.bmp")
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

            #print(f"Saved HDR image to {hdr_filename}")
        cv2.destroyAllWindows()

""""""
def align_images(im1, im2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Create FLANN matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Store good matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography to warp image
    height, width, channels = im2.shape
    im1_aligned = cv2.warpPerspective(im1, H, (width, height))

    return im1_aligned

# Load images
image1 = cv2.imread('path_to_your_reference_image.jpg')
image2 = cv2.imread('path_to_your_image_to_align.jpg')

# Align image2 to image1
aligned_image = align_images(image1, image2)

# Save or show the result
cv2.imwrite('aligned_image.jpg', aligned_image)
cv2.imshow('Aligned Image', aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()""""""



######### These Main functions need to be edited to that they act the way that we want them to. 
# the goal for this file is for it to be a utility function file.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='The root directory of the dataset')
    parser.add_argument('--path_no_fog', type=str, required=True, help='The path to the images without fog')
    args = parser.parse_args()

    make_dir_hdr(args.dataroot)

    create_hdr_images(args.dataroot, out_dataroot=args.path_no_fog, HDR_type='Debevec')

######### These Main functions need to be edited to that they act the way that we want them to. 
# the goal for this file is for it to be a utility function file.

'''if __name__ == '__main__':
    file_path_1 = './datasets/stereofog_images/final_project_dataset/B'
    file_path_2 = './datasets/stereofog_images/final_project_dataset/A'
    new_dir = './datasets/stereofog_images/final_project_dataset/no_match2'
    #dataroot = '.\\post_hdr\\final\\A'
    #identifier = '04-20'
    #rename_matching_files(dataroot, split_identifier=identifier)
    print(check_match(file_path_1, file_path_2, new_dir))'''
"""

