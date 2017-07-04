# Copyright: (C) 2017 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Claudio Fantacci
#
# idl_servervisualservoing.thrift

/**
 * ServerVisualServoingIDL
 *
 * IDL Interface to \ref ServerVisualServoing functionalities.
 */

service ServerVisualServoingIDL
{
    /**
     * Initialize the robot to an initial position.
     * The initial positions are stored on an external file and are referenced
     * by a unique label.
     *
     * @param label a label referring to one of the available initial positions;
     *              the string shall be one of the available modes returned
     *              by the get_info() method.
     *
     * @return true upon success, false otherwise.
     */
    bool stored_init(1:string label);

    /**
     * Set the robot visual servoing goal.
     * The goals are stored on an external file and are referenced by a unique
     * label.
     *
     * @param label a label referring to one of the available goals;
     *              the string shall be one of the available modes returned
     *              by the get_info() method.
     *
     * @return true upon success, false otherwise.
     */
    bool stored_go_to_goal(1:string label);

    /**
     * Get goal point from SFM module. The point is taken by clicking on a
     * dedicated 'yarpview' GUI and the orientation is hard-coded.
     *
     * @note This service is experimental and should be used with care.
     *
     * @return true upon success, false otherwise.
     */
    bool get_goal_from_sfm();

    /**
     * Gently close the visual servoing server, deallocating resources.
     */
    bool quit();



    /* NEW METHODS FROM IVISUALSERVOING */
    /**
    * Set the goal points on both left and right camera image plane and start
    * visual servoing.
    *
    * @param px_l a 8D vector which contains the (u,v) coordinates of the pixels
    *             within the left image plane. Pixels are to be provided
    *             sequentially, i.e. (u1, v1, u2, v2, ...).
    * @param px_r a 8D vector which contains the (u,v) coordinates of the pixels
    *             within the right image plane. Pixels are to be provided
    *             sequentially, i.e. (u1, v1, u2, v2, ...).
    *
    * @return true/false on success/failure.
    */
    bool go_to_point_goal(1: list<double> px_l, 2: list<double> px_r);

    /**
    * Set the goal points on both left and right camera image plane and start
    * visual servoing.
    *
    * @param vec_px_l a collection of four 2D vectors which contains the (u,v)
    *                 coordinates of the pixels within the left image plane.
    * @param vec_px_r a collection of four 2D vectors which contains the (u,v)
    *                 coordinates of the pixels within the right image plane.
    *
    * @return true/false on success/failure.
    */
    bool go_to_plane_goal(1: list<list<double>> vec_px_l, 2: list<list<double>> vec_px_r);

    /**
     *  Set visual servoing operating mode between:
     *  1. 'position': position-only visual servo control;
     *  2. 'orientation': orientation-only visual servo control;
     *  3. 'pose': position + orientation visual servo control.
     *
     * @param mode a label referring to one of the three operating mode, i.e.
     *             'position', 'orientation' or 'pose'.
     *
     * @return true/false on success/failure.
     */
    bool set_modality(1:string mode);

    /**
     * Set the point controlled during visual servoing.
     *
     * @param point label of the point to control.
     *
     * @return true/false on success/failure.
     *
     * @note The points available to control are identified by a distinct,
     *       unique label. Such labels can are stored in the bottle returned by
     *       the getInfo() method.
     */
    bool set_control_point(1: string point);

    /**
     * Return useful information for visual servoing.
     *
     * @return All the visual servoing information.
     */
    list<string> get_visual_servoing_info();

    /**
     * Set visual servoing goal tolerance.
     *
     * @param tol the tolerance in pixel.
     */
    bool set_go_to_goal_tolerance(1: double tol);

    /**
     * Check once whether the visual servoing controller is running or not.
     *
     * @return true/false on it is running/not running.
     *
     * @note The visual servoing controller may be terminated due to many
     *       different reasons, not strictly related to reaching the goal.
     */
    bool check_visual_servoing_controller();

    /**
     * Wait until visual servoing reaches the goal.
     * [wait for reply]
     *
     * @param period the check time period [s].
     * @param timeout the check expiration time [s]. If timeout <= 0 (as by
     *                default) the check will be performed without time
     *                limitation.
     *
     * @return true for success, false for failure and timeout expired.
     *
     * @note The tolerance to which the goal is considered achieved can be set
     *       with the method setGoToGoalTolerance().
     */
    bool wait_visual_servoing_done(1: double period, 2: double timeout);

    /**
     * Ask for an immediate stop of the visual servoing controller.
     * [wait for reply]
     *
     * @return true/false on success/failure.
     *
     * @note Default value: period = 0.5, timeout = 0.0
     */
    bool stop_controller();

    /**
     * Set the translation gain of the visual servoing control algorithm.
     *
     * @return true/false on success/failure.
     *
     * @note Warning: higher values of the gain corresponds to higher
     *       translatinal velocities.
     *       Default value: k_x = 0.5
     */
    bool set_translation_gain(1: double K_x);

    /**
     * Set the maximum translation velocity of the visual servoing control
     * algorithm (same for each axis).
     *
     * @param max_x_dot the maximum allowed velocity for x, y, z coordinates
     *                  [m/s].
     *
     * @return true/false on success/failure.
     */
    bool set_max_translation_velocity(1: double max_x_dot);

    /**
     * Set the orientation gain of the visual servoing control algorithm.
     *
     * @return true/false on success/failure.
     *
     * @note Warning: higher values of the gain corresponds to higher
             translatinal velocities.
             Default value: 0.5.
     */
    bool set_orientation_gain(1: double K_o);

    /**
     * Set the maximum angular velocity of the axis-angle velocity vector of the
     * visual servoing control algorithm.
     *
     * @param max_x_dot the maximum allowed angular velocity [rad/s].
     *
     * @return true/false on success/failure.
     */
    bool set_max_orientation_velocity(1: double max_o_dot);

    /**
     * Helper function: extract four Cartesian points lying on the plane defined
     * by the frame o in the position x relative to the robot base frame.
     *
     * @param x a 3D vector which is filled with the actual position x,y,z [m].
     * @param o a 4D vector which is filled with the actual orientation using
     *          axis-angle representation xa, ya, za, theta [rad].
     *
     * @return a collection of four Cartesian points (position only) extracted
     *         by the plane defined by x and o.
     */
    list<list<double>> get_3D_position_goal_from_3D_pose(1: list<double> x, 2: list<double> o);

    /**
     * Helper function: extract four 2D pixel points lying on the plane defined
     * by the frame o in the position x relative to the robot base frame.
     *
     * @param x a 3D vector which is filled with the actual position x,y,z [m].
     * @param o a 4D vector which is filled with the actual orientation using
     *          axis-angle representation xa, ya, za, theta [m]/[rad].
     * @param cam either "left" or "right" to select left or right camera.
     *
     * @return a collection of three Cartesian points (position only) extracted
     *         by the plane defined by x and o.
     */
    list<list<double>> get_pixel_position_goal_from_3D_pose(1: list<double> x, 2: list<double> o, 3: string cam);

}
