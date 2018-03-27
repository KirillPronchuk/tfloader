import cv2 

def grayscale(img):    
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:,:,None]

def normalize(value):
    return value / 255 * 2 - 1

def preprocess_image(image):
    img = grayscale(image)
    img = normalize(img)
    return img

from PIL import Image
jpgfile = Image.open("30km.jpg")
print preprocess_image(jpgfile)

