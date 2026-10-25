#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class CameraView:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('eye_in_hand_camera_view', anonymous=True)
        rospy.Subscriber('/eye_in_hand/image_raw', Image, self.image_callback)
        rospy.loginfo("Eye-in-hand camera viewer started.")
        rospy.spin()

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("Eye-in-Hand RGB Camera", frame)
            cv2.waitKey(1)
        except CvBridgeError as error:
            rospy.logerr(str(error))

if __name__ == '__main__':
    try:
        CameraView()
    finally:
        cv2.destroyAllWindows()

