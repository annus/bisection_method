import os
import sys
import time
import shutil
import threading
import pyautogui as pg
from PIL import Image
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
        resized_image = shot.crop((400, 100, 1030, 280))
        resized_image.save(os.path.join('game_folder', 'image_'+str(i)+'.jpg'))
        verbose = 'Taken {} screenshots, with a delay of {} seconds'.format(i, 1./float(sys.argv[2]))
        print(verbose, end='', flush=True)
        print('\b'*len(verbose), end='', flush=True)
    print('total screenshots=', i)
main()









