#!/usr/bin/env python3
"""
Red-cylinder localiser for an eye-in-hand camera (ROS1 / Gazebo classic).

Publishes /cylinder/pose (PoseStamped in the world frame):
  position    = cylinder centre (x, y, z_plane)
  orientation = yaw of the cylinder's long axis (roll = pitch = 0)

Improvements over the first version:
  * TF is looked up at the IMAGE timestamp (not Time(0)) -> no sync error
  * plane height is a parameter and depends on lying / standing
  * axis direction from image moments, back-projected to the plane
  * median over N frames -> removes pixel noise, gives a std for validation
  * red mask handles the HSV hue wrap-around (0..10 and 170..180)

Params (private):  ~radius ~length ~lying ~avg_frames ~world_frame
"""
import math
import collections

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_from_euler, quaternion_matrix


class CylinderLocator:
    def __init__(self):
        self.world = rospy.get_param("~world_frame", "world")
        self.radius = rospy.get_param("~radius", 0.010)   # MEASURE from your SDF
        self.length = rospy.get_param("~length", 0.06)     # MEASURE from your SDF
        self.lying = rospy.get_param("~lying", True)
        # False = image frame is a ROS link frame (x fwd, y left, z up), e.g. frameName=camera_link
        self.optical_frame = rospy.get_param("~optical_frame", False)
        n = rospy.get_param("~avg_frames", 15)

        # height of the cylinder centre above the table (table assumed at z=0)
        self.z_plane = self.radius if self.lying else self.length / 2.0

        self.bridge = CvBridge()
        self.K = None
        self.buf = collections.deque(maxlen=n)
        self.tfbuf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfbuf)

        rospy.Subscriber("/eye_in_hand/camera_info", CameraInfo, self.info_cb, queue_size=1)
        rospy.Subscriber("/eye_in_hand/image_raw", Image, self.img_cb, queue_size=1)
        self.pub = rospy.Publisher("/cylinder/pose", PoseStamped, queue_size=1)
        self.dbg = rospy.Publisher("/cylinder/debug_image", Image, queue_size=1)

    # ------------------------------------------------------------------
    def info_cb(self, msg):
        self.K = np.array(msg.K, dtype=float).reshape(3, 3)

    @staticmethod
    def red_mask(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (0, 120, 40), (10, 255, 255)) | \
            cv2.inRange(hsv, (170, 120, 40), (180, 255, 255))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def ray_to_plane(self, u, v, T_wc):
        """Intersect the pixel ray with z = z_plane. T_wc: 4x4 camera->world."""
        ray_c = np.linalg.inv(self.K) @ np.array([u, v, 1.0])  # optical: x right, y down, z fwd
        if not self.optical_frame:
            ray_c = np.array([ray_c[2], -ray_c[0], -ray_c[1]])  # -> x fwd, y left, z up
        ray_w = T_wc[:3, :3] @ ray_c
        origin = T_wc[:3, 3]
        if abs(ray_w[2]) < 1e-6:
            return None
        t = (self.z_plane - origin[2]) / ray_w[2]
        return None if t <= 0 else origin + t * ray_w

    # ------------------------------------------------------------------
    def img_cb(self, msg):
        if self.K is None:
            return
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        mask = self.red_mask(bgr)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            self.buf.clear()
            return
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 80:
            return
        M = cv2.moments(c)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        # orientation of the principal axis in the image
        th = 0.5 * math.atan2(2 * M["mu11"], M["mu20"] - M["mu02"])

        # camera pose at the moment the image was taken
        try:
            tf = self.tfbuf.lookup_transform(self.world, msg.header.frame_id,
                                             msg.header.stamp, rospy.Duration(0.1))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            return
        q = tf.transform.rotation
        T = quaternion_matrix([q.x, q.y, q.z, q.w])
        T[:3, 3] = [tf.transform.translation.x, tf.transform.translation.y,
                    tf.transform.translation.z]

        p0 = self.ray_to_plane(cx, cy, T)
        p1 = self.ray_to_plane(cx + 30 * math.cos(th), cy + 30 * math.sin(th), T)
        if p0 is None or p1 is None:
            return
        yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        # axis has a 180 deg ambiguity -> store the doubled angle
        self.buf.append((p0[0], p0[1], math.cos(2 * yaw), math.sin(2 * yaw)))

        if len(self.buf) == self.buf.maxlen:
            a = np.array(self.buf)
            x, y = np.median(a[:, 0]), np.median(a[:, 1])
            yaw = 0.5 * math.atan2(a[:, 3].mean(), a[:, 2].mean())
            out = PoseStamped()
            out.header.frame_id = self.world
            out.header.stamp = msg.header.stamp
            out.pose.position.x, out.pose.position.y, out.pose.position.z = x, y, self.z_plane
            qq = quaternion_from_euler(0, 0, yaw)
            (out.pose.orientation.x, out.pose.orientation.y,
             out.pose.orientation.z, out.pose.orientation.w) = qq
            self.pub.publish(out)
            rospy.loginfo_throttle(1.0, "cyl x=%.4f y=%.4f yaw=%.1fdeg  std_xy=(%.1f,%.1f)mm",
                                   x, y, math.degrees(yaw),
                                   1e3 * a[:, 0].std(), 1e3 * a[:, 1].std())

        dbg = bgr.copy()
        cv2.drawContours(dbg, [c], -1, (0, 255, 0), 2)
        cv2.circle(dbg, (int(cx), int(cy)), 4, (255, 0, 0), -1)
        self.dbg.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))


if __name__ == "__main__":
    rospy.init_node("cylinder_world_position")
    CylinderLocator()
    rospy.spin()
