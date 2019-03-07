from mss import mss
import cv2


def screen():
    with mss() as sct:
        sct.shot()
    im = cv2.imread('C:/Users/ddarkreaper/OneDrive/Work - MetricsFlow/Wheely/monitor-1.png')
    r = im(:,:,1)


while True:
    screen()

