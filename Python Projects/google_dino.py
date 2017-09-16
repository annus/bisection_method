import os
import sys
import time
import shutil
import threading
import pyautogui as pg
from PIL import Image
from curtsies import Input
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam
import h5py
i = 0

def play():
    print('\b', end='', flush=True)
    pg.press('space')   

def take_screenshot(time_del=0.5):
    threading.Timer(time_del, take_screenshot).start()
    global i
    i += 1
    shot = pg.screenshot()
    resized_image = shot.crop((400, 100, 1030, 280))
    resized_image.save(os.path.join('game_folder', 'image_'+str(i)+'.jpg'))

def main():
    if os.path.exists('game_folder'): 
        shutil.rmtree('game_folder')
        time.sleep(1)
    os.mkdir('game_folder')
    # take_screenshot(time_del=(1./float(sys.argv[2])))
    time_end = time.time()+float(sys.argv[1])
    while time.time() <= time_end:
        # play()  
        global i
        i += 1
        shot = pg.screenshot()
        croped_image = shot.crop((400, 100, 1030, 280))
        croped_image = croped_image.resize((80,80), Image.ANTIALIAS)
        croped_image.save(os.path.join('game_folder', 'image_'+str(i)+'.jpg'))
        verbose = ' Taken {} screenshots'.format(i)
        print(verbose, end='', flush=True)
        print('\b'*len(verbose), end='', flush=True)
    print('Stats:\ndelay: {} seconds\naverage: \
{} shots per second'.format(float(sys.argv[1])/i, 
    int(1./(float(sys.argv[1])/i))))
# main()

def conv_model(verbose=False):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), input_shape=(3, 80, 80), 
        strides=(4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (4, 4), input_shape=(32, 80, 80), 
        strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), input_shape=(64, 80, 80), 
        strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(3))

    adam = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    model.save('new_model_instance.h5')
    if verbose: print('log: Model saved!') 
    return model

new_model = conv_model(verbose=True)


 
















