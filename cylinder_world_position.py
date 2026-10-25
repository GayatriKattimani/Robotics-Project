#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import tf

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CylinderWorldPosition:

    def __init__(self):
        rospy.init_node('cylinder_world_position')

        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()

        # Camera intrinsic parameters from camera_info
        self.fx = 554.3827128
        self.fy = 554.3827128
        self.cx = 320.5
        self.cy = 240.5

        self.image_sub = rospy.Subscriber(
            '/eye_in_hand/image_raw',
            Image,
            self.image_callback,
            queue_size=1
        )

        rospy.loginfo("Cylinder world-position node started")

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
        except Exception as e:
            rospy.logerr(str(e))
            return

        # Detect the red cylinder
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 80])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return

        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) < 100:
            return

        # Estimate the centre of the red cylinder contour
        moments = cv2.moments(contour)

        if moments['m00'] == 0:
            return

        u = moments['m10'] / moments['m00']
        v = moments['m01'] / moments['m00']

        # Estimate the dominant 2D direction of the cylinder
        vx, vy, x0, y0 = cv2.fitLine(
            contour,
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        ).flatten()

        angle = np.degrees(np.arctan2(vy, vx))
               
        # Draw the detected contour and fitted axis
        display = image.copy()

        cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)

        line_length = 100
        p1 = (
            int(x0 - vx * line_length),
            int(y0 - vy * line_length)
        )
        p2 = (
            int(x0 + vx * line_length),
            int(y0 + vy * line_length)
        )

        cv2.line(display, p1, p2, (255, 0, 0), 2)
        cv2.circle(display, (int(u), int(v)), 5, (0, 255, 255), -1)

        cv2.imshow("Cylinder axis detection", display)
        cv2.waitKey(1)

        rospy.loginfo_throttle(
            1,
            "Cylinder image centre = (%.1f, %.1f), "
            "projected axis angle = %.1f deg",
            u, v, angle
        )
        
        # Get camera pose in the world frame
        try:
            translation, quaternion = self.tf_listener.lookupTransform(
                'world',
                'camera_link',
                rospy.Time(0)
            )
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException
        ):
            rospy.logwarn_throttle(2, "Waiting for camera TF...")
            return

        # Pixel ray in optical coordinates:
        # x = right, y = down, z = forward
        ray_optical = np.array([
            (u - self.cx) / self.fx,
            (v - self.cy) / self.fy,
            1.0
        ])

        # Convert to camera_link coordinates.
        # Assumes camera_link: x forward, y left, z up.
        ray_camera = np.array([
            ray_optical[2],
            -ray_optical[0],
            -ray_optical[1]
        ])

        # Rotate the ray into world coordinates
        rotation = tf.transformations.quaternion_matrix(
            quaternion
        )[:3, :3]

        ray_world = np.dot(rotation, ray_camera)
        rospy.loginfo_throttle(1, "Ray in world frame: %s", np.round(ray_world, 3))
        camera_world = np.array(translation)

        # Intersect the ray with the ground plane z = 0
        if abs(ray_world[2]) < 1e-6:
            rospy.logwarn_throttle(
                2, "Camera ray is parallel to ground"
            )
            return

        cylinder_height = 0.025
        t = (cylinder_height - camera_world[2]) / ray_world[2]

        if t <= 0:
            rospy.logwarn_throttle(
                2, "Ground plane is behind the camera ray"
            )
            return

        cylinder_world = camera_world + t * ray_world

        rospy.loginfo_throttle(
            1,
            "Whole cylinder centroid = (%.1f, %.1f) | "
            "Ground-plane projection = (%.3f, %.3f, 0.000) m",
            u,
            v,
            cylinder_world[0],
            cylinder_world[1]
        )


if __name__ == '__main__':
    try:
        node = CylinderWorldPosition()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

