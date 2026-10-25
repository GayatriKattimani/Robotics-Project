
#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()

def image_callback(msg):
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red has two ranges in HSV because hue wraps around
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 80])
    upper_red2 = np.array([179, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of detected red regions
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Select the largest red region
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        if area > 100:
            # Calculate the center using image moments
            M = cv2.moments(contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw the contour and center
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                cv2.putText(
                    frame, f"Center: ({cx}, {cy})",
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )

                rospy.loginfo_throttle(
                    2, f"Cylinder center: u={cx}, v={cy}"
                )

    # Display the processed image and mask
    cv2.imshow("Cylinder Detection", frame)
    cv2.imshow("Red Mask", mask)
    cv2.waitKey(1)


rospy.init_node("cylinder_detection")

rospy.Subscriber(
    "/eye_in_hand/image_raw", Image, image_callback
)

cv2.namedWindow("Cylinder Detection", cv2.WINDOW_NORMAL)
cv2.namedWindow("Red Mask", cv2.WINDOW_NORMAL)

rospy.spin()

cv2.destroyAllWindows()

