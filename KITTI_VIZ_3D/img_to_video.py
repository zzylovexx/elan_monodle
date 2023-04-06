import cv2
import os
def image_to_video():
    file = './video1/'  
    output = 'img_toivdeo.mp4'  
    height = 375
    weight = 1242
    fps = 5
    # fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G') 用于avi格式的生成
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    videowriter = cv2.VideoWriter(output, fourcc, fps, (weight, height))  # 创建一个写入视频对象
    for file_name  in sorted(os.listdir(file)):
        path = file + file_name
        print(path)
        frame = cv2.imread(path)
        videowriter.write(frame)

    videowriter.release()

image_to_video()
