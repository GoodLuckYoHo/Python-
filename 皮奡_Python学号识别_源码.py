
import cv2
import math

import numpy as np
from scipy import ndimage



def show(image, window_name):
    cv2.namedWindow(window_name, 0)
    cv2.imshow(window_name, image)
    # 0任意键终止窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#输入图片并进行调整
img = cv2.imread('H:/sample/sample2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
 
# 霍夫变换进行校正
lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
rotate_angle = 0
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    if x1 == x2 or y1 == y2:
        continue
    t = float(y2 - y1) / (x2 - x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle

image = ndimage.rotate(img,rotate_angle)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#图像转灰度
gaosi =cv2.GaussianBlur(gray, (1,1), 0)
#高斯平滑
blur = cv2.medianBlur(gaosi, 7)
#图像过滤
threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#二值处理
canny = cv2.Canny(threshold, 100, 200)
#边缘检测
kernel = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(canny, kernel, iterations=5)
#为了使边缘检测的边缘更加连贯，使用膨胀处理，对白色的边缘膨胀，即边缘线条变得更加粗一些。
contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_copy = image.copy()
res = cv2.drawContours(image_copy, contours, -1, (255, 0, 0), 20)
#使用findContours对边缘膨胀过的图片进行轮廓检测
contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
image_copy = image.copy()
res = cv2.drawContours(image_copy, contours, -1, (255, 0, 0), 20)
#经过对轮廓的面积排序，我们可以准确的提取出校园卡的轮廓。



epsilon = 0.02 * cv2.arcLength(contours, True)

approx = cv2.approxPolyDP(contours, epsilon, True)
n = []
for x, y in zip(approx[:, 0, 0], approx[:, 0, 1]):
    n.append((x, y))
n = sorted(n)
sort_point = []
n_point1 = n[:2]
n_point1.sort(key=lambda x: x[1])
sort_point.extend(n_point1)
n_point2 = n[2:4]
n_point2.sort(key=lambda x: x[1])
n_point2.reverse()
sort_point.extend(n_point2)
p1 = np.array(sort_point, dtype=np.float32)
h = sort_point[1][1] - sort_point[0][1]
w = sort_point[2][0] - sort_point[1][0]
pts2 = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)

# 生成变换矩阵
M = cv2.getPerspectiveTransform(p1, pts2)
# 进行透视变换
dst = cv2.warpPerspective(image, M, (w, h))
# print(dst.shape)
#提取出轮廓的四个顶点，并按顺序进行排序，对所选的卡片图像进行校正处理

if w < h:
    dst = np.rot90(dst)
resize = cv2.resize(dst, (1084, 669), interpolation=cv2.INTER_AREA)

#图像宽高检测变换，以确定图像是否为正
temp_image = resize.copy()


gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
gaosi =cv2.GaussianBlur(gray, (1,1), 0)
threshold = cv2.threshold(gaosi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
blur = cv2.medianBlur(threshold, 5)
kernel = np.ones((3, 3), np.uint8)
morph_open = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
#将图像中需要的区域显现出来。
kernel = np.ones((7, 7), np.uint8)
dilate = cv2.dilate(morph_open, kernel, iterations=6)
contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
resize_copy = resize.copy()
res = cv2.drawContours(resize_copy, contours, -1, (255, 0, 0), 2)

labels = ['']
positions = []
data_areas = {}
resize_copy = resize.copy()
#长方形区域框选
for contour in contours:
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    x, y, w, h = cv2.boundingRect(approx)
    if h > 50 and x < 670:
        res = cv2.rectangle(resize_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        area = gray[y:(y + h), x:(x + w)]
        blur = cv2.medianBlur(area, 3)
        data_area = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        positions.append((x, y))
        data_areas['{}-{}'.format(x, y)] = data_area

crop_img = res[206:265,432:665]


 
gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
ret1,gray=cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

blur = cv2.medianBlur(threshold, 5)
canny1 = cv2.Canny(blur, 100, 150)
kernel = np.ones((3, 3), np.uint8)
morph_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
#将图像中需要的区域显现出来。
kernel = np.ones((7, 7), np.uint8)
dilate = cv2.dilate(morph_open, kernel, iterations=1)
contours, hierarchy = cv2.findContours(canny1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

res1 = cv2.drawContours(crop_img, contours, -1, (255, 0, 0), 1)
print(len(contours))
for c in contours:
    x1,y1,w1,h1=cv2.boundingRect(c)
    caiji=cv2.rectangle(threshold,(x1,y1),(x1+w1,y1+h1),(255,255,255),1)

cv2.imshow("caiji",caiji)
cv2.waitKey()   #要加的两行代码
cv2.destroyAllWindows()


#对模板进行处理
test_img = cv2.imread('H:/sample/modetest.jpg')
graytest = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
ret,graytest=cv2.threshold(graytest, 200, 255, cv2.THRESH_BINARY)
thresholdtest = cv2.threshold(graytest, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

blurtest = cv2.medianBlur(thresholdtest, 5)
cannytest = cv2.Canny(blurtest, 100, 150)
morphtest_open = cv2.morphologyEx(cannytest, cv2.MORPH_OPEN, kernel)
dilatetest = cv2.dilate(morphtest_open, kernel, iterations=1)
contourstest, hierarchytest = cv2.findContours(cannytest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
res2 = cv2.drawContours(test_img, contourstest, -1, (255, 0, 0), 1)

print(len(contourstest))
for c in contourstest:
    x1,y1,w1,h1=cv2.boundingRect(c)
    muban=cv2.rectangle(thresholdtest,(x1,y1),(x1+w1,y1+h1),(0,255,255),1)
    
muban1=255-muban




def sequence_contours(image,width,height):

    contours,hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)
    RectBoxes0 = np.ones((n,4),dtype=int)
    for i in range(n):
        RectBoxes0[i] = cv2.boundingRect(contours[i])
    
    RectBoxes = np.ones((n,4),dtype=int)
    for i in range(n):
        sequence = 0
        for j in range(n):
            if RectBoxes0[i][0]>RectBoxes0[j][0]:
                sequence = sequence + 1
        RectBoxes[sequence]=RectBoxes0[i]
    ImgBoxes = [[]for i in range(n)]
    for i in range(n):
            x,y,w,h = RectBoxes[i]
            ROI = image[y:y+h,x:x+w]
            ROI = cv2.resize(ROI,(width,height))
            thresh_val, ROI = cv2.threshold(ROI, 200, 255, cv2.THRESH_BINARY)
            ImgBoxes[i] = ROI
            
    return RectBoxes,ImgBoxes
       
        
    

RectBoxes_temp,ImgBoxes_temp = sequence_contours(graytest,120,200)
print(RectBoxes_temp)
cv2.imshow('ImgBoxes_temp[1]', ImgBoxes_temp[3])
cv2.waitKey()   #要加的两行代码
cv2.destroyAllWindows()
RectBoxes, ImgBoxes = sequence_contours(caiji, 120, 200)
print(RectBoxes)

result = []
for i in range(len(ImgBoxes)):
    score = np.zeros(len(ImgBoxes_temp),dtype=int)
   
    for j in range(len(ImgBoxes_temp)):
        
        score[j] = cv2.matchTemplate(ImgBoxes[i], ImgBoxes_temp[j], cv2.TM_SQDIFF)
        
    min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(score)
    result.append(min_indx[1])
print(result)

result = str(result)
im=res
cv2.putText(im,result,(451, 199),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,255),2)

cv2.imshow("final", im)
cv2.waitKey()   #要加的两行代码
cv2.destroyAllWindows()
    













