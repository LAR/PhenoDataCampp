from ij.gui import NewImage
from ij import IJ

# get the current image selected in Fiji
imp = IJ.getImage()
processor = imp.getProcessor()

# get width and height of image
w = processor.getWidth()
h = processor.getHeight()

# make new image (initially filled with black pixels)
output = NewImage.createImage("Dilated",w, h, 1, 8, NewImage.FILL_BLACK)
output_processor = output.getProcessor()

# search all neighbouring pixels of each pixel in turn
for x in range(1, w-1):
	for y in range(1, h-1):
		foundWhite = False
		for a in range(x-1, x+2):
			for b in range(y-1, y+2):
				if(processor.getPixel(a,b)==255):
					foundWhite = True

		if foundWhite:
			output_processor.setColor(255)
			output_processor.drawPixel(x,y)

# show the result
output.show()


