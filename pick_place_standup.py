import math

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import GetModelState
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest

# ---------------- geometry taken from your xacro ----------------
YAW_AXIS_X = 0.012
SHOULDER_Z = 0.0765
A = (0.024, 0.128)
LA = math.hypot(*A)
BETA = math.atan2(A[1], A[0])
LB = 0.124
LE = 0.126
MARGIN = 0.03
LIMITS = [(-0.9 * math.pi, 0.9 * math.pi),
          (-0.57 * math.pi, 0.5 * math.pi),
          (-0.3 * math.pi, 0.44 * math.pi),
          (-0.57 * math.pi, 0.65 * math.pi)]
OBS_POSE = [0.0, 0.0, 0.0, 1.57]
DOWN, FORWARD = math.pi / 2, 0.0


def ik(x, y, z, phi):
    """World target for end_effector_link. phi = tool pitch. Returns [q1..q4] or None."""
    dx, dy = x - YAW_AXIS_X, y
    r = math.hypot(dx, dy)
    q1 = math.atan2(dy, dx)
    r4 = r - LE * math.cos(phi)
    z4 = z + LE * math.sin(phi)
    ex, ez = r4, z4 - SHOULDER_Z
    c = (ex * ex + ez * ez - LA * LA - LB * LB) / (2 * LA * LB)
    if abs(c) > 1.0:
        return None
    delta = -math.acos(c)
    th1 = math.atan2(ez, ex) - math.atan2(LB * math.sin(delta), LA + LB * math.cos(delta))
    th2 = th1 + delta
    q2 = BETA - th1
    q3 = -th2 - q2
    q4 = phi - q2 - q3
    q = [q1, q2, q3, q4]
    for qi, (lo, hi) in zip(q, LIMITS):
        if not (lo + MARGIN <= qi <= hi - MARGIN):
            return None
    return q


class Truth:
    """Gazebo ground truth (used for checks and diagnostics only, never to plan the grasp)."""
    def __init__(self, name):
        self.name, self.srv = name, None

    def pose(self):
        try:
            if self.srv is None:
                rospy.wait_for_service("/gazebo/get_model_state", 2.0)
                self.srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
            r = self.srv(self.name, "world")
            return r.pose if r.success else None
        except (rospy.ROSException, rospy.ServiceException):
            return None


class Arm:
    def __init__(self, dry, ee_frame, refine):
        self.dry, self.ee_frame, self.refine = dry, ee_frame, refine
        self.js = {}
        self.joint_srv = rospy.ServiceProxy("/goal_joint_space_path", SetJointPosition)
        self.tool_srv = rospy.ServiceProxy("/goal_tool_control", SetJointPosition)
        self.tfbuf = tf2_ros.Buffer()
        if not dry:
            rospy.wait_for_service("/goal_joint_space_path")
            rospy.wait_for_service("/goal_tool_control")
            rospy.Subscriber("/joint_states", JointState, self._js_cb, queue_size=1)
            tf2_ros.TransformListener(self.tfbuf)

    def _js_cb(self, msg):
        self.js = dict(zip(msg.name, msg.position))

    @staticmethod
    def _req(group, names, pos, t):
        req = SetJointPositionRequest()
        req.planning_group = group
        req.joint_position.joint_name = names
        req.joint_position.position = list(pos)
        req.joint_position.max_velocity_scaling_factor = 1.0
        req.joint_position.max_accelerations_scaling_factor = 1.0
        req.path_time = t
        return req

    # ---------------- arm ----------------
    def move(self, label, q, t=3.0):
        rospy.loginfo("%-12s joints [%s]", label, ", ".join("%.3f" % v for v in q))
        if self.dry:
            return
        res = self.joint_srv(self._req("arm", ["joint1", "joint2", "joint3", "joint4"], q, t))
        if not res.is_planned:
            raise RuntimeError("controller could not plan '%s'" % label)
        rospy.sleep(t + 0.5)

    def ee_position(self):
        try:
            tr = self.tfbuf.lookup_transform("world", self.ee_frame, rospy.Time(0), rospy.Duration(1.0)).transform.translation
            return (tr.x, tr.y, tr.z)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            return None

    def move_to(self, label, target, phi, t=3.0, refine=True, iters=2, tol=0.0015):
        """IK move to a world target. With refine: measure the REAL end-effector position (TF) and shift the
        command by the error, so sag / controller offsets do not put the jaws beside the object."""
        x, y, z = target
        cx_, cy_, cz_ = x, y, z
        for i in range(iters + 1 if (refine and self.refine and not self.dry) else 1):
            q = ik(cx_, cy_, cz_, phi)
            if q is None:
                if i == 0:
                    raise RuntimeError("'%s' unreachable" % label)
                break
            self.move(label if i == 0 else label + "*", q, t if i == 0 else 1.5)
            if self.dry:
                return
            if not (refine and self.refine):
                return
            ee = self.ee_position()
            if ee is None:
                rospy.logwarn("no TF %s: cannot correct the position", self.ee_frame)
                return
            err = (x - ee[0], y - ee[1], z - ee[2])
            rospy.loginfo("             %s: EE error  x %+.1f  y %+.1f  z %+.1f mm", label,
                          err[0] * 1e3, err[1] * 1e3, err[2] * 1e3)
            if max(abs(e) for e in err) < tol:
                return
            if err[2] < -tol and max(abs(err[0]), abs(err[1])) < tol:
                return        # only z is off and the EE is HIGHER than asked: something (the ground) blocks it; pushing harder is wrong
            clip = lambda e: max(-0.015, min(0.015, e))
            cx_ += clip(err[0])
            cy_ += clip(err[1])
            cz_ = max(z, cz_ + clip(err[2]))      # compensate sag upward, never command lower than requested

    # ---------------- gripper ----------------
    def finger(self):
        return self.js.get("gripper")

    def _cmd_gripper(self, pos, t):
        res = self.tool_srv(self._req("gripper", ["gripper"], [pos], t))
        if not res.is_planned:
            rospy.logwarn("gripper service REFUSED the command %.4f", pos)

    def open_fully(self, pos, t=2.0, tol=0.002, tries=4):
        rospy.loginfo("%-12s gripper -> %.4f (fully open)", "open", pos)
        if self.dry:
            return True
        g = None
        for attempt in range(1, tries + 1):
            self._cmd_gripper(pos, t)
            rospy.sleep(t + 0.5)
            g, gs = self.js.get("gripper"), self.js.get("gripper_sub")
            rospy.loginfo("             fingers: gripper=%s gripper_sub=%s",
                          "?" if g is None else "%.4f" % g, "?" if gs is None else "%.4f" % gs)
            if g is None or (abs(g - pos) <= tol and (gs is None or abs(gs - pos) <= tol)):
                return True
            rospy.logwarn("gripper not fully open yet (attempt %d of %d)", attempt, tries)
        return False

    def close_ramp(self, end, duration, probe=None, squeeze=0.004, lag_tol=0.0015, still_tol=0.0001, still_time=1.2):
        """Close SLOWLY. The gripper service ignores path_time (the fingers jump to the target), so the ramp is made
        here: the target moves linearly from the current finger value to `end` over `duration` seconds, one small
        command every 0.4 s. If the fingers stop while the target is still ahead of them (they touched something)
        the ramp ends and the fingers press `squeeze` metres beyond the contact point.
        Returns (final finger value, 'contact' | 'closed' | 'unknown')."""
        rospy.loginfo("%-12s close SLOWLY -> %.4f over %.0f s (stops early on contact)", "close", end, duration)
        if self.dry:
            return None, "dry"
        start = self.finger()
        if start is None:
            self._cmd_gripper(end, 1.0)
            rospy.sleep(duration)
            return None, "unknown"
        t0 = rospy.Time.now()
        ref, ref_t, last_cmd, last_log = start, t0, -1.0, -1.0
        contact = False
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            el = (now - t0).to_sec()
            frac = min(1.0, el / duration)
            target = start + (end - start) * frac
            g = self.finger()
            if abs(g - ref) > still_tol:
                ref, ref_t = g, now
            if el > 0.3 and el - last_cmd >= 0.4:
                last_cmd = el
                self._cmd_gripper(target, 0.6)
            if el - last_log >= 1.0:
                last_log = el
                rospy.loginfo("             t=%2.0f s  target %.4f  finger %.4f%s", el, target, g, probe() if probe else "")
            still_for = (now - ref_t).to_sec()
            if el > 2.0 and (g - target) > lag_tol and still_for >= still_time:
                contact = True                       # the fingers stopped although the target keeps going
                break
            if frac >= 1.0 and still_for >= still_time:
                break                                # ramp finished and the fingers are at rest
            rate.sleep()
        g = self.finger()
        if contact:
            rospy.loginfo("             fingers stopped at %.4f: contact. Pressing %.1f mm further.", g, squeeze * 1e3)
            self._cmd_gripper(max(end, g - squeeze), 1.0)
            rospy.sleep(1.5)
            g = self.finger()
        return g, ("contact" if contact else "closed")

def read_cylinder(arm, dry, radius):
    """Go to the observation pose and read the vision estimate. Returns (x, y, z, axis_yaw) or None."""
    if dry:
        return (rospy.get_param("~cx", 0.21), rospy.get_param("~cy", 0.03), radius,
                rospy.get_param("~cyaw", 0.0))
    arm.move("observe", OBS_POSE, 3.0)
    rospy.sleep(1.0)
    try:
        msg = rospy.wait_for_message("/cylinder/pose", PoseStamped, timeout=15)
    except rospy.ROSException:
        published = [t for t, _ in rospy.get_published_topics()]
        if "/cylinder/pose" not in published:
            rospy.logerr("Nothing publishes /cylinder/pose: start  roslaunch cylinder_pick vision.launch")
        else:
            rospy.logerr("Vision node runs but sees no cylinder: it was probably pushed out of view. "
                         "Re-place it:  rosrun cylinder_pick place_cylinder_random.py")
        return None
    p = msg.pose
    return (p.position.x, p.position.y, p.position.z, 2.0 * math.atan2(p.orientation.z, p.orientation.w))


def _clear_private_params():
    """Params given as _name:=value STAY on the ROS master after the node exits and silently change the next run
    (e.g. _stop_after:=grasp). Remove them at exit."""
    try:
        rospy.delete_param("~")
    except Exception:
        pass


def main():
    rospy.init_node("pick_place_standup")
    rospy.on_shutdown(_clear_private_params)
    P = rospy.get_param
    dry = P("~dry_run", False)
    radius, length = P("~radius", 0.010), P("~length", 0.06)
    grip_open = P("~grip_open", 0.010)
    grip_close = max(-0.010, P("~grip_close", -0.008))
    # The fingertips sit about 16 mm below end_effector_link, so at EE z = 0.0163 they rest ON THE GROUND. Pressing
    # them down made the fingers close 5x slower (ground friction) and stop short. +8 mm keeps them just above it.
    # EE height above the object centre. The ground stops the end effector at about z = 16 mm, so for a 30 mm object
    # (centre 15 mm) use about +4 mm; for the 20 mm bar (centre 10 mm) about +9 mm. auto_raise fixes it if the tips touch.
    grasp_dz = P("~grasp_dz", 0.004)
    close_time = P("~close_time", 20.0)   # seconds for the whole closing ramp
    squeeze = P("~squeeze", 0.005)
    q_touch = P("~q_touch", -0.008)    # finger value when the jaws are closed on NOTHING (measured earlier)
    close_wait = P("~close_wait", 40.0)  # max seconds to wait for the slow fingers to stop
    stall_tol = P("~stall_tol", 0.002)
    max_attempts = 1                      # ONE try only: the grasp either passes or fails
    stop_after = P("~stop_after", "")
    place_x, place_y = P("~place_x", 0.25), P("~place_y", -0.08)
    model = P("~model", "red_cylinder")
    flip = (0.20, 0.0, 0.10)

    if dry:
        rospy.logwarn("*** DRY RUN: nothing will move. If you did not ask for this: "
                      "rosparam delete /pick_place_standup ***")
    arm = Arm(dry, P("~ee_frame", "end_effector_link"), P("~refine", True))
    truth = Truth(model)

    # fixed waypoints after the grasp: check them once
    fixed = [("flip-point", flip[0], flip[1], flip[2], DOWN),
             ("flip-wrist", flip[0], flip[1], flip[2], FORWARD),
             ("pre-place", place_x, place_y, length / 2.0 + 0.04, FORWARD),
             ("place", place_x, place_y, length / 2.0 + 0.004, FORWARD),
             ("retreat-up", place_x, place_y, length / 2.0 + 0.08, FORWARD)]
    fj = {}
    for label, x, y, z, phi in fixed:
        q = ik(x, y, z, phi)
        if q is None:
            rospy.logerr("'%s' at (%.3f, %.3f, %.3f) is unreachable. Change _place_x/_place_y.", label, x, y, z)
            return
        fj[label] = q

    held = False
    for attempt in range(1, max_attempts + 1):
        rospy.loginfo("=============== grasp (one try) ===============")
        found = read_cylinder(arm, dry, radius)
        if found is None:
            return
        cx, cy, cz, cyaw = found
        rospy.loginfo("cylinder (vision) at (%.4f, %.4f, %.4f), axis yaw %.1f deg", cx, cy, cz, math.degrees(cyaw))
        gt = None if dry else truth.pose()
        if gt is not None:
            rospy.loginfo("vision minus Gazebo truth: dx %+.1f mm  dy %+.1f mm",
                          (cx - gt.position.x) * 1e3, (cy - gt.position.y) * 1e3)

        if gt is not None and abs(cz - gt.position.z) > 0.003:
            rospy.logwarn("the vision node assumes the object centre is at z = %.1f mm but Gazebo says %.1f mm. Restart "
                          "the vision node with _radius:=%.3f (and the pick script with _radius:=%.3f)",
                          cz * 1e3, gt.position.z * 1e3, gt.position.z, gt.position.z)
        azimuth = math.atan2(cy, cx - YAW_AXIS_X)
        mis = (cyaw - azimuth + math.pi / 2) % math.pi - math.pi / 2
        rospy.loginfo("axis vs radial direction: %.1f deg", math.degrees(mis))
        if abs(mis) > math.radians(P("~align_tol_deg", 20.0)):
            rospy.logerr("Cylinder axis is %.0f deg off radial. Re-place it with place_cylinder_random.py "
                         "(or _align_tol_deg:=30).", math.degrees(mis))
            return
        hov = next((h for h in (0.06, 0.05, 0.04, 0.03) if ik(cx, cy, cz + h, DOWN) is not None), None)
        if hov is None or ik(cx, cy, cz + grasp_dz, DOWN) is None:
            rospy.logerr("Cylinder at (%.3f, %.3f) is out of reach for a top-down grasp.", cx, cy)
            return

        if not arm.open_fully(grip_open, 2.0):
            rospy.logerr("ABORT: the gripper does not open.")
            return
        arm.move_to("pre-grasp", (cx, cy, cz + hov), DOWN, 3.0)
        arm.move_to("grasp", (cx, cy, cz + grasp_dz), DOWN, 3.0)
        ee_now = None if dry else arm.ee_position()
        if ee_now is not None:
            short = (cz + grasp_dz) - ee_now[2]      # negative: the end effector could not get down to its target
            if short < -0.001:
                new_dz = grasp_dz - short + 0.003
                if P("~auto_raise", True) and ik(cx, cy, cz + new_dz, DOWN) is not None:
                    rospy.logwarn("fingertips still touch the ground (EE %.1f mm above its target): raising the grasp "
                                  "by %.1f mm", -short * 1e3, (new_dz - grasp_dz) * 1e3)
                    grasp_dz = new_dz
                    arm.move_to("grasp-up", (cx, cy, cz + grasp_dz), DOWN, 1.5, refine=False)
                else:
                    rospy.logwarn("fingertips still touch the ground: use a larger _grasp_dz (now %.0f mm)", grasp_dz * 1e3)

        # alignment of the jaws with the cylinder (ground truth): lateral must stay within the jaw clearance
        ee, gt = (None, None) if dry else (arm.ee_position(), truth.pose())
        if ee is not None and gt is not None:
            dx, dy, dz = gt.position.x - ee[0], gt.position.y - ee[1], gt.position.z - ee[2]
            rad = dx * math.cos(azimuth) + dy * math.sin(azimuth)
            lat = -dx * math.sin(azimuth) + dy * math.cos(azimuth)
            rospy.loginfo("ALIGNMENT cylinder minus EE: radial %+.1f mm | LATERAL %+.1f mm | height %+.1f mm",
                          rad * 1e3, lat * 1e3, dz * 1e3)
        z_before = gt.position.z if gt is not None else radius     # object height just before closing
        g0 = None if dry else arm.finger()
        if g0 is not None and g0 > grip_open + 0.002:
            rospy.logwarn("the jaws are FORCED APART (finger %.4f > open %.4f): the object is as wide as the "
                          "opening and is wedged between them. Use a thinner object.", g0, grip_open)
        if stop_after == "grasp":
            rospy.loginfo("stopped at the grasp pose, gripper open")
            return

        def probe():
            e, c = arm.ee_position(), truth.pose()
            if e is None or c is None:
                return ""
            dx, dy = c.position.x - e[0], c.position.y - e[1]
            return "  | cyl z %.1f mm, off EE: radial %+.0f lateral %+.0f mm" % (
                c.position.z * 1e3, (dx * math.cos(azimuth) + dy * math.sin(azimuth)) * 1e3,
                (-dx * math.sin(azimuth) + dy * math.cos(azimuth)) * 1e3)

        g_end, close_mode = arm.close_ramp(grip_close, close_time, probe=probe, squeeze=squeeze)
        q_held = q_touch + radius        # each finger travels `radius` from the touching position onto the object
        if dry:
            state = "held"
        elif close_mode == "contact":
            state = "held"               # the fingers stopped on something before reaching the end of the ramp
        elif close_mode == "unknown":
            state = "unknown"
        else:
            state = "missed"             # the fingers closed all the way: nothing was between the jaws
        if not dry:
            rospy.loginfo("grasp check: fingers ended at %s (%s) -> %s", "?" if g_end is None else "%.4f" % g_end,
                          close_mode, state.upper())
        if stop_after == "close":
            rospy.loginfo("stopped after closing (%s). Now run in ANOTHER terminal:  gz topic -e "
                          "/gazebo/default/physics/contacts | grep -E 'collision[12]:' | sort | uniq -c", state)
            return

        if state in ("missed", "not_closed") and P("~finger_check_aborts", False):
            rospy.logwarn("grasp check says '%s': skipping the lift", state)
            arm.open_fully(grip_open, 2.0)
            continue
        # By default the finger verdict is only information (q_touch may be wrong for your gripper): the
        # lift below measures with Gazebo ground truth whether the object really rose.
        arm.move_to("lift", (cx, cy, cz + hov), DOWN, 2.0, refine=False)
        if dry:
            held = True
        else:
            st = truth.pose()
            if st is not None:
                rise = st.position.z - z_before
                held = rise > 0.015
                rospy.loginfo("cylinder rose %.1f mm after the lift -> %s", rise * 1e3, "HELD" if held else "NOT held")
            else:
                g = arm.finger()
                held = g is not None and g > -0.004
                rospy.loginfo("no Gazebo truth; finger at %s -> %s", g, "held?" if held else "empty")
        if stop_after == "lift":
            rospy.loginfo("stopped after the lift")
            return
    if held:
        rospy.loginfo("GRASP PASSED: the cylinder was lifted")
    else:
        rospy.logerr("GRASP FAILED: the cylinder was not lifted. Continuing with the full motion anyway "
                     "(flip, place position, retreat, back to the observation pose).")

    arm.move("flip-point", fj["flip-point"], 3.0)
    arm.move("flip-wrist", fj["flip-wrist"], 3.0)
    arm.move("pre-place", fj["pre-place"], 3.0)
    arm.move("place", fj["place"], 2.0)
    arm.open_fully(grip_open, 2.0)
    arm.move("retreat-up", fj["retreat-up"], 2.0)
    arm.move("observe", OBS_POSE, 3.0)

    if not dry:
        from tf.transformations import quaternion_matrix
        st = truth.pose()
        if st is None:
            return
        R = quaternion_matrix([st.orientation.x, st.orientation.y, st.orientation.z, st.orientation.w])
        tilt = math.degrees(math.acos(min(1.0, abs(R[2, 2]))))
        ok = tilt < 10 and abs(st.position.z - length / 2.0) < 0.01
        rospy.loginfo("final: pos (%.3f, %.3f, %.3f), tilt %.1f deg -> %s", st.position.x, st.position.y,
                      st.position.z, tilt, "PLACE PASSED (standing upright)" if ok else "PLACE FAILED (not standing)")
        rospy.loginfo("=========== RESULT: grasp %s | place %s ===========",
                      "PASSED" if held else "FAILED", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    main()
