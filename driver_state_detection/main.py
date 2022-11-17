import time

import cv2
import dlib
import numpy as np

from Utils import get_face_area
from Eye_Dector_Module import EyeDetector as EyeDet
from Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from Attention_Scorer_Module import AttentionScorer as AttScorer

# capture source number select the webcam to use (default is zero -> built in camera)
CAPTURE_SOURCE = 0

# camera matrix obtained from the camera calibration script, using a 9x6 chessboard
camera_matrix = np.array(
    [[899.12150372, 0., 644.26261492],
     [0., 899.45280671, 372.28009436],
     [0, 0,  1]], dtype="double")

# distortion coefficients obtained from the camera calibration script, using a 9x6 chessboard
dist_coeffs = np.array(
    [[-0.03792548, 0.09233237, 0.00419088, 0.00317323, -0.15804257]], dtype="double")


def main():

    ctime = 0  # current time (used to compute FPS)
    ptime = 0  # past time (used to compute FPS)
    prev_time = 0  # previous time variable, used to set the FPS limit
    fps_lim = 10  # FPS upper limit value, needed for estimating the time for each frame and increasing performances
    time_lim = 1. / fps_lim  # time window for each frame taken by the webcam

    cv2.setUseOptimized(True)  # set OpenCV optimization to True

    # instantiation of the dlib face detector object
    Detector = dlib.get_frontal_face_detector()
    Predictor = dlib.shape_predictor(
        "predictor/shape_predictor_68_face_landmarks.dat")  # instantiation of the dlib keypoint detector model
    '''
    the keypoint predictor is compiled in C++ and saved as a .dat inside the "predictor" folder in the project
    inside the folder there is also a useful face keypoint image map to understand the position and numnber of the
    various predicted face keypoints
    '''

    # instantiation of the eye detector and pose estimator objects
    Eye_det = EyeDet(show_processing=False)

    Head_pose = HeadPoseEst(show_axis=True)

    # instantiation of the attention scorer object, with the various thresholds
    # NOTE: set verbose to True for additional printed information about the scores
    Scorer = AttScorer(fps_lim, ear_tresh=0.15, ear_time_tresh=2, gaze_tresh=0.2,
                       gaze_time_tresh=2, pitch_tresh=35, yaw_tresh=28, pose_time_tresh=2.5, verbose=False)
    
    # capture the input from the default system camera (camera number 0)
    cap = cv2.VideoCapture(CAPTURE_SOURCE)

    gazeCounter = 0
    counter = 0
    perclos_tresh = 0.15
    warnState = False
    level_two_warning = fps_lim*90

    if not cap.isOpened():  # if the camera can't be opened exit the program
        print("Cannot open camera")
        exit()
    
    while True:  # infinite loop for webcam video capture

        delta_time = time.perf_counter() - prev_time  # delta time for FPS capping
        ret, frame = cap.read()  # read a frame from the webcam

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

         # if the frame comes from webcam, flip it so it looks like a mirror.
        if CAPTURE_SOURCE == 0:
            frame = cv2.flip(frame, 2)

        if delta_time >= time_lim:  # if the time passed is bigger or equal than the frame time, process the frame
            prev_time = time.perf_counter()

            # compute the actual frame rate per second (FPS) of the webcam video capture stream, and show it
            ctime = time.perf_counter()
            fps = 1.0 / float(ctime - ptime)
            ptime = ctime
            cv2.putText(frame, "FPS:" + str(round(fps,0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,(255, 0, 255), 1)

            # transform the BGR frame in grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # apply a bilateral filter to lower noise but keep frame details
            gray = cv2.bilateralFilter(gray, 5, 10, 10)

            # find the faces using the dlib face detector
            faces = Detector(gray)

            if len(faces) > 0:  # process the frame only if at least a face is found

                # take only the bounding box of the biggest face
                faces = sorted(faces, key=get_face_area, reverse=True)
                driver_face = faces[0]

                # predict the 68 facial keypoints position
                landmarks = Predictor(gray, driver_face)

                # shows the eye keypoints (can be commented)
                Eye_det.show_eye_keypoints(color_frame=frame, landmarks=landmarks)

                # compute the EAR score of the eyes
                ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)

                # compute the PERCLOS score and state of tiredness
                tired, perclos_score = Scorer.get_PERCLOS(ear)

                # compute the Gaze Score
                gaze = Eye_det.get_Gaze_Score(frame=gray, landmarks=landmarks)

                # compute the head pose
                frame_det, roll, pitch, yaw = Head_pose.get_pose(frame=frame, landmarks=landmarks)

                # if the head pose estimation is successful, show the results
                if frame_det is not None:
                    frame = frame_det

                # show the real-time EAR score
                if ear is not None:
                    cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

                # show the real-time Gaze Score
                if gaze is not None:
                    cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 80),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

                # show the real-time PERCLOS score
                cv2.putText(frame, "PERCLOS:" + str(round(perclos_score, 3)), (10, 110),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
                
                # if the driver is tired, show and alert on screen
                if tired:  
                    cv2.putText(frame, "TIRED!", (10, 280),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                # evaluate the scores for EAR, GAZE and HEAD POSE
                asleep, looking_away, distracted = Scorer.eval_scores(
                    ear, gaze, roll, pitch, yaw)  

                # if the state of attention of the driver is not normal, show an alert on screen
                if asleep:
                    cv2.putText(frame, "ASLEEP!", (10, 300),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                if looking_away:
                    cv2.putText(frame, "LOOKING AWAY!", (10, 320),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                if distracted:
                    cv2.putText(frame, "DISTRACTED!", (10, 340),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                
                if tired or perclos_score>= perclos_tresh:
                    counter+=1
                    if counter>=level_two_warning:
                        warnState = True
                    print("careful, you may be tired!")
                if gaze is None:
                    if gazeCounter>=fps_lim:
                        gazeCounter=0
                        print("danger, eyes closed!")
                    gazeCounter+=1
                if warnState:
                    print("please stop driving immediately! you are tired.")
            

            cv2.imshow("Frame", frame)  # show the frame on screen

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return


#FUNCTIONS NOT TO BE USED YET, TESTING PHASE
'''
def indication():
    # compute the PERCLOS score and state of tiredness
    fps_lim = 25
    tired, perclos_score = Scorer.get_PERCLOS(ear)
    gazeCounter = 0
    counter = 0
    gaze = Eye_det.get_Gaze_Score(frame=gray, landmarks=landmarks)
    #running inside the loop
    warnState = counter==2
    if tired or perclos_score>= 0.15:
        counter+=1
        print("careful, you may be tired!")
    if gaze is None:
        if gazeCounter>=fps_lim:
            gazeCounter=0
            print("danger, eyes closed!")
        gazeCounter+=1
    if warnState:
        print("please stop driving immediately! you are tired.")
    return

def lip_distance(landmarks):

    #prerequisites
    Detector = dlib.get_frontal_face_detector()
    Predictor = dlib.shape_predictor("predictor/shape_predictor_68_face_landmarks.dat")
    faces = Detector(gray)
    faces = sorted(faces, key=get_face_area, reverse=True)
    driver_face = faces[0]

    # predict the 68 facial keypoints position
    landmarks = Predictor(gray, driver_face)

    #running inside the loop
    top_lip = landmarks[50:53]
    top_lip = np.concatenate((top_lip, landmarks[61:64]))

    low_lip = landmarks[56:59]
    low_lip = np.concatenate((low_lip, landmarks[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

def yawn_detection():
    #also another way to do this is a yawn threshold of 20

    #outside of the loop
    prev_distance = lip_distance(landmarks)
    start_yawn_timer = 0.0
    stop_yawn_timer = 0.0
    start_stop = False
    #recommended to close your mouth at the start of the program so the original distance is almost none
    original_distance = lip_distance(landmarks)

    #running inside the loop
    #checking if lips are opening
    if lip_distance(landmarks)>prev_distance:

        #checking the start of the yawn
        if (prev_distance == original_distance) and not start_stop:
            start_yawn_timer = time.perf_counter
            start_stop = True

        #checking the end of the yawn 
        if(prev_distance == original_distance) and start_stop:
            stop_yawn_timer = time.perf_counter
            start_stop = False
        
        #if start_stop is false, then the yawn must be done and we calculate the time of the yawn, checking if it was greater than 4s
        if not start_stop:
            if (stop_yawn_timer-start_yawn_timer)>4:
                print("You yawned!")
'''

if __name__ == "__main__":
    main()