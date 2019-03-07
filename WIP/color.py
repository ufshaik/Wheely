from PIL import Image

im = Image.open('dead_parrot.jpg') # Can be many different formats.
pix = im.load()
print im.size  # Get the width and hight of the image for iterating over
print pix[x,y]