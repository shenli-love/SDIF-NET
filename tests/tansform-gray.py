import cv2

image = cv2.imread('D:/machinelearn/infrared_and_visible_image_fusion/Tar/SDIF-net/datasets/M3FD_Detection/vi/00333.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('D:/machinelearn/infrared_and_visible_image_fusion/Tar/SDIF-net/tests/00333_gray.png', image_gray)
cv2.imshow('image', image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()