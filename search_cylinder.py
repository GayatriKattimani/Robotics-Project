import math

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_matrix

YAW_AXIS_X = 0.012                        # joint1 axis is 0.012 m ahead of the world origin
SCAN_BANDS = [(0.0, 0.0, 1.57),           # (q2, q3, q4): camera straight down, sees x ~0.16-0.27
              (0.0, 0.0, 1.20)]           # camera tilted forward, sees x ~0.23-0.38


def ray_to_plane(K, R_wc, origin, u, v, z_plane):
    """Pixel -> point on the plane z = z_plane. Camera frame = ROS link frame (x fwd, y left, z up)."""
    ray_opt = np.linalg.inv(K) @ np.array([u, v, 1.0])        # optical: x right, y down, z fwd
    ray_link = np.array([ray_opt[2], -ray_opt[0], -ray_opt[1]])
    ray_w = R_wc @ ray_link
    if abs(ray_w[2]) < 1e-6:
        return None
    t = (z_plane - origin[2]) / ray_w[2]
    return None if t <= 0 else origin + t * ray_w


class Searcher:
    def __init__(self):
        self.z_plane = rospy.get_param("~radius", 0.010)
        self.q1_min = rospy.get_param("~q1_min", -1.5)
        self.q1_max = rospy.get_param("~q1_max", 1.5)
        self.q1_step = rospy.get_param("~q1_step", 0.25)
        self.min_area = rospy.get_param("~min_area", 150)

        self.bridge = CvBridge()
        self.K = None
        self.img = None
        self.tfbuf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfbuf)
        rospy.Subscriber("/eye_in_hand/camera_info", CameraInfo, self.info_cb, queue_size=1)
        rospy.Subscriber("/eye_in_hand/image_raw", Image, self.img_cb, queue_size=1)
        self.pub = rospy.Publisher("/cylinder/coarse_pose", PoseStamped, queue_size=1, latch=True)

        rospy.wait_for_service("/goal_joint_space_path")
        self.joint_srv = rospy.ServiceProxy("/goal_joint_space_path", SetJointPosition)

    def info_cb(self, msg):
        self.K = np.array(msg.K, dtype=float).reshape(3, 3)

    def img_cb(self, msg):
        self.img = msg

    # ------------------------------------------------------------------ motion
    def move(self, q, t=2.0):
        req = SetJointPositionRequest()
        req.planning_group = "arm"
        req.joint_position.joint_name = ["joint1", "joint2", "joint3", "joint4"]
        req.joint_position.position = list(q)
        req.joint_position.max_velocity_scaling_factor = 1.0
        req.joint_position.max_accelerations_scaling_factor = 1.0
        req.path_time = t
        if not self.joint_srv(req).is_planned:
            raise RuntimeError("controller could not plan the move to %s" % q)
        rospy.sleep(t + 0.7)                      # let the arm settle before looking

    # ------------------------------------------------------------------ vision
    def fresh_image(self, timeout=3.0):
        """Wait for an image taken AFTER now, so motion blur / old frames are not used."""
        t0 = rospy.Time.now()
        end = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < end:
            if self.img is not None and self.img.header.stamp > t0:
                return self.img
            rospy.sleep(0.05)
        return None

    def detect(self):
        """Return (msg, u, v, area, touches_border) for the biggest red blob, or None."""
        msg = self.fresh_image()
        if msg is None:
            rospy.logwarn("no fresh image - is the simulation running?")
            return None
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 120, 40), (10, 255, 255)) | \
               cv2.inRange(hsv, (170, 120, 40), (180, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            return None
        m = cv2.moments(c)
        u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
        x, y, w, h = cv2.boundingRect(c)
        H, W = mask.shape
        border = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
        return msg, u, v, area, border

    def coarse_world(self, msg, u, v):
        if self.K is None:
            return None
        try:
            tf = self.tfbuf.lookup_transform("world", msg.header.frame_id,
                                             msg.header.stamp, rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            return None
        q = tf.transform.rotation
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        o = np.array([tf.transform.translation.x, tf.transform.translation.y,
                      tf.transform.translation.z])
        return ray_to_plane(self.K, R, o, u, v, self.z_plane)

    # ------------------------------------------------------------------ search
    def run(self):
        while self.K is None and not rospy.is_shutdown():
            rospy.loginfo_throttle(2, "waiting for camera_info ...")
            rospy.sleep(0.2)
        rospy.sleep(1.0)                          # let TF fill up

        q1_values = np.arange(self.q1_min, self.q1_max + 1e-6, self.q1_step)
        for band, (q2, q3, q4) in enumerate(SCAN_BANDS):
            rospy.loginfo("=== scan band %d: camera pitch joint4 = %.2f ===", band + 1, q4)
            for q1 in q1_values:
                self.move([q1, q2, q3, q4])
                hit = self.detect()
                rospy.loginfo("joint1 = %+.2f rad : %s", q1,
                              "RED OBJECT" if hit else "nothing")
                if not hit:
                    continue
                return self.report_and_aim(q1, (q2, q3, q4), hit)
        rospy.logwarn("Cylinder not found in either band. Is it inside the arm's reach "
                      "(about x = 0.12-0.27 m) and on the table?")
        return None

    def report_and_aim(self, q1, band, hit):
        msg, u, v, area, border = hit
        p = self.coarse_world(msg, u, v)
        if p is None:
            rospy.logerr("found red pixels at (%.0f, %.0f) but could not get a world position (TF?)", u, v)
            return None
        # turn the base toward the cylinder, then look again for a better coarse estimate
        azimuth = math.atan2(p[1], p[0] - YAW_AXIS_X)
        rospy.loginfo("coarse estimate (%.3f, %.3f); aiming joint1 at %.3f rad", p[0], p[1], azimuth)
        self.move([azimuth, band[0], band[1], band[2]])
        again = self.detect()
        if again:
            msg, u, v, area, border = again
            p2 = self.coarse_world(msg, u, v)
            if p2 is not None:
                p = p2
        if border:
            rospy.logwarn("blob touches the image border: part of the cylinder is cut off, "
                          "the estimate is rough")
        out = PoseStamped()
        out.header.frame_id = "world"
        out.header.stamp = rospy.Time.now()
        out.pose.position.x, out.pose.position.y, out.pose.position.z = p
        out.pose.orientation.w = 1.0
        self.pub.publish(out)
        rospy.loginfo("FOUND: coarse position x = %.3f, y = %.3f (centre height assumed %.3f), "
                      "blob area %d px, pixel (%.0f, %.0f)", p[0], p[1], self.z_plane, area, u, v)
        reach = math.hypot(p[0] - YAW_AXIS_X, p[1])
        if reach > 0.26:
            rospy.logwarn("distance from the base is %.2f m: beyond a top-down grasp (about 0.25 m). "
                          "Move the cylinder closer.", reach)
        return p


if __name__ == "__main__":
    rospy.init_node("search_cylinder")
    try:
        Searcher().run()
    except rospy.ROSInterruptException:
        pass
