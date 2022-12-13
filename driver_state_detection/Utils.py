import numpy as np
import cv2
import flet as f


def resize(frame, scale_percent):
    """
    Resize the image maintaining the aspect ratio
    :param frame: opencv image/frame
    :param scale_percent: int
        scale factor for resizing the image
    :return:
    resized: rescaled opencv image/frame
    """
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
    return resized


def get_face_area(face):
    """
    Computes the area of the bounding box ROI of the face detected by the dlib face detector
    It's used to sort the detected faces by the box area

    :param face: dlib bounding box of a detected face in faces
    :return: area of the face bounding box
    """
    return abs((face.left() - face.right()) * (face.bottom() - face.top()))


def show_keypoints(keypoints, frame):
    """
    Draw circles on the opencv frame over the face keypoints predicted by the dlib predictor

    :param keypoints: dlib iterable 68 keypoints object
    :param frame: opencv frame
    :return: frame
        Returns the frame with all the 68 dlib face keypoints drawn
    """
    
    x = keypoints.part(42).x
    y = keypoints.part(42).y
    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
    x = keypoints.part(39).x
    y = keypoints.part(39).y
    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
    x = keypoints.part(36).x
    y = keypoints.part(36).y
    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
    x = keypoints.part(45).x
    y = keypoints.part(45).y
    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

    #Around the face
    x = keypoints.part(1).x
    y = keypoints.part(1).y
    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
    x = keypoints.part(15).x
    y = keypoints.part(15).y
    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
    x = keypoints.part(8).x
    y = keypoints.part(8).y
    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
    x = keypoints.part(30).x
    y = keypoints.part(30).y
    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
    x = keypoints.part(12).x
    y = keypoints.part(12).y
    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1) 
    x = keypoints.part(4).x
    y = keypoints.part(4).y
    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1) 

    #mouth
    x = keypoints.part(48).x
    y = keypoints.part(48).y
    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1) 
    x = keypoints.part(54).x
    y = keypoints.part(54).y
    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1) 
    return


def midpoint(p1, p2):
    """
    Compute the midpoint between two dlib keypoints

    :param p1: dlib single keypoint
    :param p2: dlib single keypoint
    :return: array of x,y coordinated of the midpoint between p1 and p2
    """
    return np.array([int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)])


def get_array_keypoints(landmarks, dtype="int", verbose: bool = False):
    """
    Converts all the iterable dlib 68 face keypoint in a numpy array of shape 68,2

    :param landmarks: dlib iterable 68 keypoints object
    :param dtype: dtype desired in output
    :param verbose: if set to True, prints array of keypoints (default is False)
    :return: points_array
        Numpy array containing all the 68 keypoints (x,y) coordinates
        The shape is 68,2
    """
    points_array = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        points_array[i] = (landmarks.part(i).x, landmarks.part(i).y)

    if verbose:
        print(points_array)

    return points_array

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

def add_alert(page, lv, msg):
    lv.controls.append(f.Text(msg))
    page.update()