import cv2
import numpy as np


def keep_special_color(image, up_bound, down_bound):
    """
    保留某种特定的颜色
    :param image:
    :param up_bound:
    :param down_bound:
    :return:
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 先腐蚀后膨胀会去除孤立的小噪点
    mask = cv2.inRange(hsv_image, down_bound, up_bound)
    dilate_mask = cv2.dilate(mask, None, iterations=1)
    erode_mask = cv2.erode(dilate_mask, None, iterations=1)

    masked_image = cv2.bitwise_and(image, image, mask=erode_mask)

    return masked_image


def y_channel_equalize_hist(image):
    """
    进行 y 通道直方图均衡化。
    主要目的是加强颜色对比度。
    ** 查看效果后决定暂时不用。
    :param image:
    :return:
    """
    # Y 通道均衡化
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    yuv_channels = cv2.split(yuv_image)
    yuv_channels[0] = cv2.equalizeHist(yuv_channels[0])
    channels = cv2.merge(yuv_channels)

    result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)

    return result


def normalize_hist(image):
    """
    直方图正规化
    :param image:
    :return:
    """
    norm_img = cv2.normalize(image, dst=None, alpha=350, beta=10, norm_type=cv2.NORM_MINMAX)

    return norm_img


def sharpen_image(image):
    """
    对图像进行锐化
    :param image:
    :return:
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    result = cv2.filter2D(image, -1, kernel=kernel)

    return result


def remove_redundancy_color(image):
    """
    去除图片中不想要的颜色。
    目前暂定去掉除蓝、绿、青以外的颜色。
    :param image:
    :return:
    """
    blue_up_bound = np.array([124, 255, 255])
    blue_down_bound = np.array([100, 43, 46])

    cyan_up_bound = np.array([99, 255, 255])
    cyan_down_bound = np.array([78, 43, 46])

    green_up_bound = np.array([77, 255, 255])
    green_down_bound = np.array([35, 43, 46])

    blue_image = keep_special_color(image, blue_up_bound, blue_down_bound)
    cyan_image = keep_special_color(image, cyan_up_bound, cyan_down_bound)
    green_image = keep_special_color(image, green_up_bound, green_down_bound)

    tmp_image = cv2.add(blue_image, cyan_image)
    tmp_image = cv2.add(tmp_image, green_image)

    result = y_channel_equalize_hist(tmp_image)

    result = sharpen_image(result)

    return result


def resize_image(image, fx, fy):
    """
    对图像进行缩放。以减少计算量。
    :param image:
    :param fx:
    :param fy:
    :return:
    """
    if not (fx and fy):
        raise ValueError("fx and fy can't be zero at the same time in resize_image() !")
    if (0 < fy < 1 < fx) or (0 < fx < 1 < fy):
        raise ValueError("fx and fy must be bigger than one at same time or smaller than one at same time"
                         " in resize_image() !")

    height, width = image.shape[:2]

    if fx > 1:
        fx = fx / width
    if fy > 1:
        fy = fy / height

    if fx == -1:
        fx = fy
    if fy == -1:
        fy = fx

    resized_image = cv2.resize(image, (0, 0), fx=fx, fy=fy)

    print("original w and h:", width, height)
    print("resize w and h:", resized_image.shape[1], resized_image.shape[0])
    cv2.imshow("original", image)
    cv2.imshow("resize", resized_image)
    cv2.waitKey(0)

    return resized_image
