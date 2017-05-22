# Copyright: (C) 2017 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Claudio Fantacci
#
# idl_visualsirpf.thrift

/**
 * VisualSIRParticleFilterIDL
 *
 * IDL Interface to \ref VisualSIRParticleFilter options.
 */

service VisualSIRParticleFilterIDL
{
    /**
     * Use/Don't use the analog values from the right hand to correct the finger
     * poses.
     *
     * @param status true/false to use/don't use analog values.
     *
     * @return true activation/deactivation success, false otherwise.
     */
    bool use_analogs(1:bool status);

    /**
     * Get information about recursive Bayesian filter, like it's status, the
     * available methods, and the current one in use, to extract the state
     * estimate from the particle set.
     *
     * @return a string with all the available information, 'none' otherwise
     */
    list<string> get_info();

    /**
     * Cheange the current method to extract the state estimate from the
     * particle set.
     *
     * @param status a string with the state estimete extraction method to use;
     *               the string shall be one of the available methods returned
     *               by the get_info() method.
     *
     * @return true method change success, false otherwise.
     */
    bool set_estimates_extraction_method(1:string method);

    /**
     * Enable/Disable the visual correction step of the SIR Particle Filter.
     *
     * @param status true/false to enable/disable visual correction
     *
     * @return true activation/deactivation success, false otherwise.
     *
     * @note By disabling the visual correction step the particles will be only
     *       propageted over time with the prediction step.
     */
    bool visual_correction(1:bool status)

    /**
     * Gently close the application deallocating resources.
     */
    bool quit();
}
