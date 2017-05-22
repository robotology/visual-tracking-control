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
     * Get information about the visual servoing server, like it's status, the
     * available initial positions, the available goals and any other available
     * option.
     *
     * @return a string with all the available information, 'none' otherwise
     */
    list<string> get_info();

    /**
     * Initialize the robot to an initial position.
     *
     * @param label a label referring to one of the available initial positions;
     *              the string shall be one of the available modes returned
     *              by the get_info() method.
     *
     * @return true upon success, false otherwise.
     */
    bool init(1:string label);

    /**
     * Set the robot visual servoing goal.
     *
     * @param label a label referring to one of the available goal positions;
     *              the string shall be one of the available modes returned
     *              by the get_info() method.
     *
     * @return true upon success, false otherwise.
     */
    bool set_goal(1:string label);

    /**
     * Get goal point from SFM module. The point is taken by clicking on a
     * dedicated 'yarpview' GUI and the orientation is hard-coded.
     *
     * @note This service is experimental and should be used with care.
     *
     * @return true upon success, false otherwise.
     */
    bool get_sfm_points();

    /**
     * Start the visual servoing controller.
     *
     * @note This is a non-blocking function.
     *
     * @return true upon success, false otherwise.
     */
    bool go();

    /**
     * Gently close the application deallocating resources.
     */
    bool quit();
}
