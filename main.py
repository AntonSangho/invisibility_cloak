import cv2
import numpy as np
import time, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', help='Input video path')
args = parser.parse_args()

# 1.VideoCapture함수를 이용해서 비디오를 불러온다.
cap = cv2.VideoCapture(args.video if args.video else 0)
# 2.카메라가 켜지는데 시간이 걸리므로 3초간 워밍업이 필요함
time.sleep(3)

# Grap background image from first part of the video
# 사람이 없는 이미지를 촬영합니다. (60프레임정도)
for i in range(60):
  ret, background = cap.read()

# 동영상을 저장하기 위한 코드 
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# 출력 비디오 파일의 이름, 코덱, 프레임 속도 및 프레임 크기를 설정합니다.
out = cv2.VideoWriter('videos/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (background.shape[1], background.shape[0]))
out2 = cv2.VideoWriter('videos/original.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (background.shape[1], background.shape[0]))

# while을 사용해서 카메라가 열려있는지를 확인합니다. 
while(cap.isOpened()):
  # 캠으로 읽어들인것을 img에 저장한다. 
  ret, img = cap.read()
  if not ret:
    break
  
  # Convert the color space from BGR to HSV
  # HSV로 변경하는 이유는 사람이 인식하는데는 HSV가 가장 비슷하기 때문이다. 
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # Generate mask to detect red color
  lower_red = np.array([0, 120, 70])
  upper_red = np.array([10, 255, 255])
  mask1 = cv2.inRange(hsv, lower_red, upper_red)

  lower_red = np.array([130, 120, 70])
  upper_red = np.array([180, 255, 255])
  mask2 = cv2.inRange(hsv, lower_red, upper_red)

  mask1 = mask1 + mask2

  # lower_black = np.array([0, 0, 0])
  # upper_black = np.array([255, 255, 80])
  # mask1 = cv2.inRange(hsv, lower_black, upper_black)


  '''
  # Refining the mask corresponding to the detected red color
  https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
  '''
  # Remove noise
  mask_cloak = cv2.morphologyEx(mask1, op=cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=2)
  mask_cloak = cv2.dilate(mask_cloak, kernel=np.ones((3, 3), np.uint8), iterations=1)
  mask_bg = cv2.bitwise_not(mask_cloak)

  cv2.imshow('mask_cloak', mask_cloak)

  # Generate the final output
  res1 = cv2.bitwise_and(background, background, mask=mask_cloak)
  res2 = cv2.bitwise_and(img, img, mask=mask_bg)
  # addWeighted로 합쳐준다. 
  result = cv2.addWeighted(src1=res1, alpha=1, src2=res2, beta=1, gamma=0)

  cv2.imshow('res1', res1)

  # cv2.imshow('ori', img)
  cv2.imshow('result', result)
  out.write(result)
  out2.write(img)

  if cv2.waitKey(1) == ord('q'):
    break

out.release()
out2.release()
cap.release()
