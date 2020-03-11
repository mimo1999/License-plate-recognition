import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

dir_path = 'D:\inputs\solved'
images = []
for filename in os.listdir(dir_path):
    img = cv2.imread(os.path.join(dir_path, filename))
    if img is not None:
        temp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        temp_resized = cv2.resize(temp, (20, 20), interpolation=cv2.INTER_CUBIC)
        images.append(temp)

img = cv2.imread('D:\inputs\Hyundai-i10-524532c.jpg_0000_0260_0540_0209_0070.png')
#img = cv2.imread('D:\inputs\Hyundai-i20-527245c.jpg_0000_0560_0448_0129_0107.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_pres = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#img_gray = cv2.equalizeHist(img_gray)
#kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#plate = cv2.filter2D(img_gray, -1, kernel)
img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
img_gray = cv2.filter2D(img_gray, -1, kernel)
a, imgThresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#imgContours, contours, npaHierarchy = cv2.findContours(imgThresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#imgContours, contours, npaHierarchy = cv2.findContours(imgThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image, contours, hierarchy = cv2.findContours(imgThresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

cont = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
i = 0
while True:
    if cv2.contourArea(cont[i])>img.shape[0]*img.shape[1]*0.9:
        i+=1
    if cv2.contourArea(cont[i])<img.shape[0]*img.shape[1]*0.3:
        break
    break
cv2.drawContours(img_pres, cont, i, (0, 255, 0), 1)
print(cv2.contourArea(cont[0]))
c = cont[i]

idx = i # The index of the contour that surrounds your object
mask = np.zeros_like(img_pres) # Create mask where white is what we want, black otherwise
cv2.drawContours(mask, cont, idx, (255, 255, 255), -1) # Draw filled contour in mask
outer = np.zeros_like(img) # Extract out the object and place into output image
print(outer.shape)
print(img.shape, 'org')
print(img_gray.shape)
outer[mask == 255] = img[mask == 255]

# Now crop
#(y, x) = np.where(mask == 255)
#(topy, topx) = (np.min(y), np.min(x))
#(bottomy, bottomx) = (np.max(y), np.max(x))
#out = out[topy:bottomy+1, topx:bottomx+1]
print(img.shape[0]*img.shape[1])
print(outer.shape)
out = cv2.cvtColor(outer, cv2.COLOR_RGB2GRAY)
print(out.shape)
plt.imshow(out)
plt.show()
a, imgThresh = cv2.threshold(out, 100, 255, cv2.THRESH_BINARY_INV)
prev_count = -1
x_crop = 0
x_crop_end = 0
for i in range(out.shape[1]):
    count=0
    for j in range(out.shape[0]):
        if out[j][i]!=0:
            count+=1
    if count > 10:
        if  abs(prev_count-count) < 4:
            x_crop = i
            break
        prev_count=count
prev_count = -1
for i in range(out.shape[1]-1, 0, -1):
    count=0
    for j in range(out.shape[0]):
        if out[j][i]!=0:
            count+=1
    if count > 10:
        if  abs(prev_count-count) < 4:
            x_crop_end = i
            break
        prev_count=count
y_crop_top = 0
y_crop_bot = 0


img_new = out[:,x_crop:x_crop_end]

end = img_new.shape[1]-1

front = np.transpose(np.nonzero(img_new[:, 1]))
back = np.transpose(np.nonzero(img_new[:, end]))
print(img_new.shape)
print(front)

left = (front[len(front)-1]+front[0])/2
right = (back[len(back)-1]+back[0])/2

#cv2.circle(img_new, (end, right), 3, (255, 0, 0), -1)
#cv2.circle(img_new, (0, left), 3, (255, 0, 0), -1)
b = img_new.shape[1]
h = abs(left - right)
mid_b = b/2
mid_h = (left+right)/2
AngleInDeg = 0
if left>right:
    AngleInRad = abs(math.atan(h / b))*-1
    AngleInDeg = AngleInRad * (180.0 / math.pi)
    R = cv2.getRotationMatrix2D((mid_b, mid_h), AngleInDeg, 0.6)
else:
    AngleInRad = abs(math.atan(h / b))
    AngleInDeg = AngleInRad * (180.0 / math.pi)
    R = cv2.getRotationMatrix2D((mid_b, mid_h), AngleInDeg, 0.6)


new = cv2.warpAffine(img_new, R, (img_new.shape[1], img_new.shape[0]))

new = new**1.7
info = np.amax(new)# Get the information of the incoming image type
data = new / info # normalize the data to 0 - 1
data = 255 * data # Now scale by 255
new = data.astype(np.uint8)
print(type(img_new[0][0]), 'hrer')
print(type(new[0][0]), 'hrer')
new = np.array(new, dtype=np.uint8)
a, imgThresh = cv2.threshold(new, 150, 255, cv2.THRESH_BINARY_INV)

image, contours, hierarchy = cv2.findContours(imgThresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cont = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
rect = cv2.boundingRect(cont[1])

[X, Y, Width, Height] = rect
cv2.rectangle(new, (X, Y), (X+Width, Y+Height), (255, 255, 0), 2)
final_turn = new[Y:Y+Height, X:X+Width]
cv2.imshow('win', new)
cv2.waitKey()
cv2.destroyAllWindows()
MIN_RATIO = 0.1
MAX_RATIO = 2.2

ANGLE_MAX = 15.0


def checkIfPossibleChar(possibleChar):
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and possibleChar.intBoundingRectArea < MAX_PIXEL_AREA and
            possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
            MIN_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_RATIO and possibleChar.intBoundingRectHeight < MAX_PIXEL_HEIGHT and
            possibleChar.intBoundingRectY > Y_BORDER_PADDING and possibleChar.intBoundingRectX > X_BORDER_PADDING):
        return True
    else:
        return False

def findNearestChar(char, possibleChar):
    first = None
    second = None
    first_dist = 10000
    second_dist = 10000
    for ch in possibleChar:
        dist = abs(char.intCenterX - ch.intCenterX)
        if dist== 0:
            continue
        if dist<first_dist:
            second_dist = first_dist
            second = first
            first_dist = dist
            first = ch
        elif dist>first_dist and dist<second_dist:
            second_dist = dist
            second = ch
    return first, second
def NearestRep(char, possibleChar):
    first = None
    first_dist = 10000
    for ch in possibleChar:
        dist = abs(char.intCenterX - ch.intCenterX)
        vary = abs(char.intCenterY - ch.intCenterY)
        ar_vary = abs(char.intBoundingRectArea - ch.intBoundingRectArea)
        if dist == 0:
            if vary == 0 and ar_vary == 0:
                continue
        if dist < first_dist:
            first_dist = dist
            first = ch
    return first
def checkRepetation(possibleChars):
    rectifiedList = []
    for char in possibleChars:
        firstChar = NearestRep(char, possibleChars)
        b1 = float(abs(char.intCenterX - firstChar.intCenterX))
        h1 = float(abs(char.intCenterY - firstChar.intCenterY))
        print(char.intBoundingRectArea, firstChar.intBoundingRectArea)
        if b1>plate.shape[1]*0.2:
            continue
        if b1<5:
            if char.intBoundingRectArea > firstChar.intBoundingRectArea:
                rectifiedList.append(char)
        else:
            rectifiedList.append(char)
    return rectifiedList

def checkForAngle(possibleChars):
    rectifiedList = []
    for char in possibleChars:
        firstChar, secondChar = findNearestChar(char, possibleChars)
        b1 = float(abs(char.intCenterX - firstChar.intCenterX))
        h1 = float(abs(char.intCenterY - firstChar.intCenterY))
        b2 = float(abs(char.intCenterX - secondChar.intCenterX))
        h2 = float(abs(char.intCenterY - secondChar.intCenterY))

        if b1 == 0.0:
            AngleInRad1 = 1.5708
        else:
            AngleInRad1 = math.atan(h1 / b1)
        if b2 == 0.0:
            AngleInRad2 = 1.5708
        else:
            AngleInRad2 = math.atan(h2 / b2)

        AngleInDeg1 = AngleInRad1 * (180.0 / math.pi)
        AngleInDeg2 = AngleInRad2 * (180.0 / math.pi)

        if AngleInDeg1<ANGLE_MAX or AngleInDeg2<ANGLE_MAX:
            rectifiedList.append(char)
    return rectifiedList

class PossibleChar:

    # constructor #################################################################################
    def __init__(self, _contour):
        self.contour = _contour

        self.Rect = cv2.boundingRect(self.contour)

        [X, Y, Width, Height] = self.Rect

        self.intBoundingRectX = X
        self.intBoundingRectY = Y
        self.intBoundingRectWidth = Width
        self.intBoundingRectHeight = Height

        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))

        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)

file_loc = 'D:/inputs/solved'
file_name = 'Hyundai-i10-523412c.jpg_0000_0271_0409_0174_0056'
if abs(AngleInDeg) < 0:
    plate1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plate = out
else:
    plate1 = final_turn
    plate = final_turn

#img = cv2.imread('D:\inputs\solved\Hyundai-i10-523412c.jpg_0000_0271_0409_0174_0056.png')

#plate = out
print(plate.shape)
''''
if plate.shape[1]<100:
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
elif plate.shape[1]/plate.shape[0] > 5.5:
    plate = cv2.GaussianBlur(plate, (7, 7), 0)
    print('C')
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
elif plate.shape[1]/plate.shape[0] > 3.5:
    plate = cv2.GaussianBlur(plate, (5, 5), 0)
    print('A')
    kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
else:
    plate = cv2.GaussianBlur(plate, (5, 5), 0)
    print('B')
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
'''
plate = cv2.GaussianBlur(plate, (5, 5), 0)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
plate = cv2.filter2D(plate, -1, kernel)

#plate = cv2.medianBlur(plate, ksize=3)
#plate = cv2.Canny(plate, 30, 150)

a, imgThresh = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow('viw', imgThresh)
cv2.waitKey()
cv2.destroyAllWindows()
MIN_PIXEL_WIDTH = plate.shape[1]*0.004
MIN_PIXEL_HEIGHT = plate.shape[0]*0.2
MIN_PIXEL_AREA = plate.shape[0]*plate.shape[1]*0.003
MAX_PIXEL_AREA = plate.shape[0]*plate.shape[1]*0.3
MAX_PIXEL_HEIGHT = plate.shape[0]*0.75
X_BORDER_PADDING = plate.shape[0]*0.01
Y_BORDER_PADDING = plate.shape[0]*0.05


listOfPossibleChars = []                        # this will be the return value
contours = []
letters = []

            # find all contours in plate
imgContours, contours, npaHierarchy = cv2.findContours(imgThresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#imgContours, contours, npaHierarchy = cv2.findContours(plate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    cv2.drawContours(plate1, contours, i, (255, 0, 0))
    box = cv2.boundingRect(contours[i])
    cv2.rectangle(plate1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
plt.imshow(plate1)
plt.show()
for contour in contours:
    # for each contour
    possibleChar = PossibleChar(contour)
    if checkIfPossibleChar(possibleChar):              # if contour is a possible char, note this does not compare to other chars (yet) . . .
        listOfPossibleChars.append(possibleChar)
        #letters.append(contour)

distanceRectified = checkRepetation(listOfPossibleChars)
sameLinePossibleChar = checkForAngle(distanceRectified)

print(len(distanceRectified))

images = []
images.append([[0, 0, 0], [0, 0, 0]])
loc = []
#sorted(letters, key=)
for i in sameLinePossibleChar:
    x = i.intBoundingRectX
    x = x-2
    y = i.intBoundingRectY
    y = y-2
    end_x = x+i.intBoundingRectWidth
    end_y = y+i.intBoundingRectHeight
    end_x = end_x-2
    end_y = end_y-2
    img = plate[y:end_y, x:end_x]
    print(img.shape)
    #cv2.circle(plate1, (int(i.intCenterX), int(i.intCenterY)), 1, (255, 0, 0), -1)
    images.append(img)
    loc.append(x)
    cv2.rectangle(plate1, (x, y), (end_x, end_y), (0, 255, 0), 1)
plt.imshow(plate1)
plt.show()
images = np.array(images)
images = images[1:]
loc = np.array(loc)
ins = loc.argsort()
image = images[ins]


resized = []
for i in image:
    print(i.shape)
    resi = cv2.resize(i, (20, 20), interpolation = cv2.INTER_CUBIC)
    resized.append(resi)
    #plt.imshow(i)
    #plt.show()

file_loc = 'D:/out/'
dir_path = os.path.dirname(os.path.realpath(os.path.join(file_loc, file_name)))
dir_path = file_loc + file_name + '/'
if os.path.exists(dir_path):
    print("directory exists")
else:
    print("directory does not exist!")
try:
    os.mkdir(dir_path)
except OSError:
    print("Creation of the directory %s failed" % dir_path)
else:
    print("Successfully created the directory %s " % dir_path)

#dir_path = os.path.dirname(os.path.realpath(file_name))

for i in range(len(images)):
    name = 'Symbol-' + str(i) + '.jpg'
    #cv2.imwrite(os.path.join(dir_path, name), resized[i])


