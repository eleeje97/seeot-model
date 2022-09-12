import cv2
from src.body import Body
from PIL import Image
import shutil


def get_keypoints(img_path):
    body_estimation = Body('model/body_pose_model.pth')

    # copy image into input/
    shutil.copy(img_path, '/data/seeot-model/openpose/input/test.jpg')

    # set img_path
    test_image = 'input/test.jpg'

    # image resize
    img = Image.open(test_image)
    img_resize = img.resize((176, 256))
    img_resize.save(test_image)

    # extract keypoints
    oriImg = cv2.imread(test_image)
    candidate, subset, keypoints_x, keypoints_y = body_estimation(oriImg)

    # print(len(candidate)) # number of keypoints
    # print(len(subset))    # number of person

    # name:keypoints_y:keypoints_x
    filename = img_path.split('/')[-1]
    with open('/data/seeot-model/dior/DATA_ROOT/fasion-annotation-test.csv', 'a') as file:
        file.write(':'.join([filename, str([int(i) for i in keypoints_x]), str([int(i) for i in keypoints_y])]))
        file.writelines('\n')

    # print([int(i) for i in keypoints_x])
    # print([int(i) for i in keypoints_y])


# get_keypoints('/data/seeot-model/02_4_full.jpg')
