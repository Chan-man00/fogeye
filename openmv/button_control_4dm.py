# Untitled - By: hokis - Fri Feb 16 2024

import sensor, image, pyb, time, mjpeg
import os

if 'recorded_images' not in os.listdir():
    os.mkdir('recorded_images')

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.VGA) #QVGA prior
sensor.skip_frames(time = 2000)
clock = time.clock()

# You have to turn automatic gain control and automatic white blance off
# otherwise they will change the image gains to undo any exposure settings
# that you put in place...
sensor.set_auto_gain(False)
#sensor.IOCTL_RESET_AUTO_FOCUS
#sensor.set_auto_whitebal(False)
# Need to let the above settings get in...
sensor.skip_frames(time = 500)

# Change this value to adjust the exposure. Try 10.0/0.1/etc.
EXPOSURE_TIME_SCALE = [0.5, 0.75, 1.5, 2.0]
#current_exposure_time_in_microseconds = sensor.get_exposure_us()
#print("Current Exposure == %d" % current_exposure_time_in_microseconds)

pin = pyb.Pin("P0", pyb.Pin.IN, pyb.Pin.PULL_UP)
counter = len(os.listdir('recorded_images')) + 1
#print(counter)

blue_led = pyb.LED(3)

# Loop forever
while(True):

    if pin.value() == 0: # Pull down. Button is btwn P0 and Ground.
        blue_led.on()
        sensor.set_auto_exposure(True)
        sensor.skip_frames(time = 500) # Wait for settings to set
        #pyb.delay(1000) # Alternate way to wait for settings to set
        current_exposure_time_in_microseconds = sensor.get_exposure_us()
        NewExp = str(sensor.get_exposure_us())
        times = open("times.txt", "a") # Creates document to store base exposure time
        times.write(NewExp+"\n") # Base exposure that will be mult by scale
        print("Current Exposure == %d" % current_exposure_time_in_microseconds)


        for i in range(len(EXPOSURE_TIME_SCALE)):

            custom_exposure = int(current_exposure_time_in_microseconds * EXPOSURE_TIME_SCALE[i])
            sensor.set_auto_exposure(False,
                                     exposure_us = custom_exposure)
            #NewExp = str(sensor.get_exposure_us())
            #times.write(NewExp+"\n")
            print("New exposure == %d" % sensor.get_exposure_us())
            sensor.skip_frames(time = 20)
            img = sensor.snapshot()

            # Saving the image
            img.save('/recorded_images/' + str(counter))
            counter += 1
            pyb.delay(60)

        blue_led.off()
        #pyb.delay(30)

        print('image saved')
        times.close()

else:
    # Do nothing
    pass


