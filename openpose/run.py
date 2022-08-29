import cv2
from src.body import Body
from PIL import Image


def get_keypoints(img_path):
    body_estimation = Body('model/body_pose_model.pth')

    # copy image into input/


    # set img_path
    test_image = 'input/test.jpg'


    # image resize
    img = Image.open(test_image)
    img_resize = img.resize((176, 256))
    img_resize.save(test_image)


    # extract keypoints
    oriImg = cv2.imread(test_image)
    candidate, subset, keypoints_x, keypoints_y = body_estimation(oriImg)

    print(len(candidate)) # number of keypoints
    print(len(subset))    # number of person
    # print(candidate)
    # print(subset)


    # name:keypoints_y:keypoints_x
    print([int(i) for i in keypoints_x])
    print([int(i) for i in keypoints_y])

    # print(len(keypoints_x))
    # print(len(keypoints_y))
