/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

/**
 * VisualServoingIDL
 *
 * IDL Interface to \ref ServerVisualServoing functionalities.
 */
service VisualServoingIDL
{
    /**
     * Initialize support modules and connections to perform a visual servoing
     * task. This method must be called before any other visual servoing
     * methods. Returns upon successful or failure setup.
     *
     * @param use_direct_kin instruct the visual servoing control to either use
     *                       direct kinematic or an estimated/refined pose of
     *                       the end-effector.
     *
     * @note Default value: false. There usually is an error in the robot
     *       direct kinematics that should be compensated to perform precise
     *       visual servoing. To this end, a recursive Bayesian estimation
     *       filter is used to compensate for this error. Such filter is
     *       initialized during initialization execution.
     *
     * @return true/false on success/failure.
     */
    bool init_facilities(1: bool use_direct_kin);

    /**
     * Reset support modules and connections to perform the current initialized
     * visual servoing task. Returns upon successful or failure setup.
     *
     * @note This method also resets the recursive Bayesian estimation filter.
     *       It may happen that the recursive Bayesian filter does not provide
     *       satisfactory pose estimation or diverges. Thus this method can be
     *       used to reset the filter.
     *
     * @return true/false on success/failure.
     */
    bool reset_facilities();

    /**
     * Stop and disconnect support modules and connections used for visual
     * servoing. This method must be called when visual servoing is no longer
     * needed or a new visual servoing task need to be initialized.
     *
     * @note This method also stops the recursive Bayesian estimation filter.
     *       Thus it is suggested to call this method every time visual servoing
     *       has been completed/interrupted to have the filter stopped and
     *       initialized again during the next init call.
     *
     * @return true/false on success/failure.
     */
    bool stop_facilities();

    /**
     * Set the goal points on both left and right camera image plane and start
     * visual servoing.
     *
     * @param vec_px_l a collection of four 2D vectors which contains the (u, v)
     *                 coordinates of the pixels within the left image plane.
     * @param vec_px_r a collection of four 2D vectors which contains the (u, v)
     *                 coordinates of the pixels within the right image plane.
     *
     * @note By invoking this method, the visual servoing goal will be reached in
     *       orientation first, then in position. This is because there may not
     *       be a feasible position solution for every possible orientation.
     *
     * @return true/false on success/failure.
     */
    bool go_to_px_goal(1: list<list<double>> vec_px_l, 2: list<list<double>> vec_px_r);

    /**
     * Set the goal point (3D for the position + 4D axis-angle for
     * the orientation) and start visual servoing.
     *
     * @param vec_x a 3D vector which contains the (x, y, z) Cartesian
     *              coordinates of the goal.
     * @param vec_o a 4D vector which contains the (x, y, z) axis and theta angle
     *              of rotation of the goal.
     *
     * @note By invoking this method, the visual servoing goal will be reached in
     *       position and orientation together with two parallel tasks.
     *
     * @return true/false on success/failure.
     */
    bool go_to_pose_goal(1: list<double> vec_x, 2: list<double> vec_o);

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
     *  Set visual servo control law between:
     *  1. 'decoupled': image-based visual servoing with decoupled position and
     *                  orientation control law, the control law was proposed
     *                  in [1];
     *  2. 'robust': image-based visual servoing with averaged image Jacobians,
     *               the control law was proposed in [2];
     *
     * @param mode a label referring to one of the three visual servo controls,
     *             i.e. 'position', 'orientation' or 'pose'.
     *
     * @note [1] C. Fantacci, G. Vezzani, U. Pattacini, V. Tikhanoff and L.
     *       Natale, "Precise markerless visual servoing on unknown objects for
     *       humanoid robot platforms", to appear.
     *       [2] E. Malis, “Improving vision-based control using efficient
     *       second-order minimization techniques”, IEEE ICRA, vol. 2, p.
     *       1843–1848, 2004.
     *
     * @return true/false on success/failure.
     */
    bool set_visual_servo_control(1:string control);

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
     *
     * @return true/false on success/failure.
     *
     * @note Default value: 15.0 [pixel].
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
     *
     * @note Default values: period 0.1 [s], timeout 0.0 [s].
     */
    bool wait_visual_servoing_done(1: double period, 2: double timeout);

    /**
     * Ask for an immediate stop of the visual servoing controller.
     * [wait for reply]
     *
     * @return true/false on success/failure.
     */
    bool stop_controller();

    /**
     * Set the translation gains of the visual servoing control algorithm. The
     * two values are used, respectively, when the end-effector is far away from
     * and close to the goal.
     *
     * @return true/false on success/failure.
     *
     * @note Warning: higher values of the gain corresponds to higher
     *       translation velocities and oscillation about the goal.
     *
     * @note Default values: K_x_1 = 1.0, K_x_2 = 0.25.
     */
    bool set_translation_gain(1: double K_x_1, 2: double K_x_2);

    /**
     * Set the maximum translation velocity of the visual servoing control
     * algorithm (same for each axis).
     *
     * @param max_x_dot the maximum allowed velocity for x, y, z coordinates
     *                  [m/s].
     *
     * @return true/false on success/failure.
     *
     * @note Default value: max_x_dot = 0.025 [m/s].
     */
    bool set_max_translation_velocity(1: double max_x_dot);

    /**
     * Set the tolerance, in pixels, at which the translation control law
     * swithces its gain value.
     *
     * @return true/false on success/failure.
     *
     * @note Default value: K_x_tol = 30.0 [pixel].
     */
    bool set_translation_gain_switch_tolerance(1: double K_x_tol);

    /**
     * Set the orientation gains of the visual servoing control algorithm. The
     * two values are used, respectively, when the end-effector is far away from
     * and close to the goal.
     *
     * @return true/false on success/failure.
     *
     * @note Warning: higher values of the gain corresponds to higher
     *       orientation velocities and oscillation about the goal.
     *
     * @note Default values: K_o_1 = 1.5, K_o_2 = 0.375.
     */
    bool set_orientation_gain(1: double K_o_1, 2: double K_o_2);

    /**
     * Set the maximum angular velocity of the axis-angle velocity vector of the
     * visual servoing control algorithm.
     *
     * @param max_x_dot the maximum allowed angular velocity [rad/s].
     *
     * @return true/false on success/failure.
     *
     * @note Default value: 5 * (PI / 180.0) [rad/s].
     */
    bool set_max_orientation_velocity(1: double max_o_dot);

    /**
     * Set the tolerance, in pixels, at which the orientation control law
     * swithces its gain value.
     *
     * @return true/false on success/failure.
     *
     * @note Default value: K_o_tol = 30.0 [pixel].
     */
    bool set_orientation_gain_switch_tolerance(1: double K_o_tol);

    /**
     * Helper function: extract four Cartesian points lying on the plane defined
     * by the frame o in the position x relative to the robot base frame.
     *
     * @param x a 3D vector which is filled with the actual position (x, y, z) [m].
     * @param o a 4D vector which is filled with the actual orientation using
     *          axis-angle representation (xa, ya, za) and (theta) [rad].
     *
     * @return on success: a collection of four Cartesian points (position only)
     *         extracted from the plane defined by x and o;
     *         on failure: an empty list.
     *
     * @note It is always suggested to check whether the returned list is empty
     *       or not and to take proper counter actions.
     */
    list<list<double>> get_3D_goal_positions_from_3D_pose(1: list<double> x, 2: list<double> o);

    /**
     * Helper function: extract four 2D pixel points lying on the plane defined
     * by the frame o in the position x relative to the robot base frame.
     *
     * @param x a 3D vector which is filled with the actual position (x, y, z) [m].
     * @param o a 4D vector which is filled with the actual orientation using
     *          axis-angle representation (xa, ya, za) and (theta) [m]/[rad].
     * @param cam either "left" or "right" to select left or right camera.
     *
     * @return on success: a collection of three (u, v) pixel points
     *         extracted from the plane defined by x and o;
     *         on failure: an empty list.
     *
     * @note It is always suggested to check whether the returned list is empty
     *       or not and to take proper counter actions.
     */
    list<list<double>> get_goal_pixels_from_3D_pose(1: list<double> x, 2: list<double> o, 3: string cam);

    /**
     * Gently close the visual servoing device, deallocating resources.
     */
    bool quit();


    /* TO BE DEPRECATED */
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
}
