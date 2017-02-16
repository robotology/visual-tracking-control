# Copyright: (C) 2017 iCub Facility - Istituto Italiano di Tecnologia
# Authors: Claudio Fantacci
#
# idl_visualsirpf.thrift

/**
 * visualSIRParticleFilterIDL
 *
 * IDL Interface to \ref visualSIRParticleFilterIDL SIR particle filter options.
 */

service visualSIRParticleFilterIDL
{
    /**
     * Enable/Disable image stream resulting from the superimposition of the
     * meshes rendered using the filtered state estimates.
     * @param status true/false to turn image stream on/off.
     * @return true activation/deactivation success, false otherwise.
     */
    bool result_images(1:bool status);

    /**
     * Lock/Unlock the particle filter on the last received input.
     * @param status true/false to lock/unlock input to the SIR particle filter.
     * @return true activation/deactivation success, false otherwise.
     */
    bool lock_input(1:bool status);

    /**
     * Use/Don't use the analog values from the right hand to correct the finger
     * poses.
     * @param status true/false to use/don't use analog values.
     * @return true activation/deactivation success, false otherwise.
     */
    bool use_analogs(1:bool status);
}
