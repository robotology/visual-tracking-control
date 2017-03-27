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
    bool stream_result(1:bool status);

    /**
     * Use/Don't use the analog values from the right hand to correct the finger
     * poses.
     * @param status true/false to use/don't use analog values.
     * @return true activation/deactivation success, false otherwise.
     */
    bool use_analogs(1:bool status);

    /**
     * Gently close the application deallocating resources
     */
    void quit();
}
