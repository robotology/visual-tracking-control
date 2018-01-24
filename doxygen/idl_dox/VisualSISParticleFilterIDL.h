// This is an automatically-generated file.
// It could get re-generated if the ALLOW_IDL_GENERATION flag is on.

#ifndef YARP_THRIFT_GENERATOR_VisualSISParticleFilterIDL
#define YARP_THRIFT_GENERATOR_VisualSISParticleFilterIDL

#include <yarp/os/Wire.h>
#include <yarp/os/idl/WireTypes.h>

class VisualSISParticleFilterIDL;


/**
 * VisualSISParticleFilterIDL
 * IDL Interface to \ref VisualSIRParticleFilter options.
 */
class VisualSISParticleFilterIDL : public yarp::os::Wire {
public:
  VisualSISParticleFilterIDL();
  /**
   * Initialize and run the visual SIR particle filter. Returns upon
   * successful or failure setup.
   * @return true/false on success/failure.
   */
  virtual bool run_filter();
  /**
   * Reset the visual SIR particle filter. Returns upon successful or failure
   * reset.
   * @return true/false on success/failure.
   */
  virtual bool reset_filter();
  /**
   * Stop and reset the SIR particle filter. This method must be called when
   * the SIR particle filter is no longer needed or a new filtering task
   * need to be initialized.
   * @return true/false on success/failure.
   */
  virtual bool stop_filter();
  /**
   * Enable/Disable skipping the filtering step specified in what_step.
   * what_step can be one of the following:
   *  1) prediction: skips the whole prediction step
   *  2) state: skips the prediction step related to the state transition
   *  3) exogenous: skips the prediction step related exogenous inputs
   *  4) correction: skips the whole correction step
   * @param what_step the step to skipping
   * @param status enable/disable skipping
   * @return true/false on success/failure.
   */
  virtual bool skip_step(const std::string& what_step, const bool status);
  /**
   * Use/Don't use the analog values from the right hand to correct the finger
   * poses.
   * @param status true/false to use/don't use analog values.
   * @return true activation/deactivation success, false otherwise.
   */
  virtual bool use_analogs(const bool status);
  /**
   * Get information about recursive Bayesian filter, like it's status, the
   * available methods, and the current one in use, to extract the state
   * estimate from the particle set.
   * @return a string with all the available information, 'none' otherwise
   */
  virtual std::vector<std::string>  get_info();
  /**
   * Change the current method to extract the state estimates from the
   * particle set.
   * @param status a string with the state estimate extraction method to use;
   *               the string shall be one of the available methods returned
   *               by the get_info() method.
   * @return true method changed successfully, false otherwise.
   */
  virtual bool set_estimates_extraction_method(const std::string& method);
  /**
   * Change the window size of mobile averages for estimates extraction.
   * @param window specifies the mobile window size.
   * @return true window size changed successfully, false otherwise.
   * @note The default value is 20. Minimum value is 2. Maximum value is 90.
   */
  virtual bool set_mobile_average_window(const int16_t window = 20);
  /**
   * Gently close the application, deallocating resources.
   */
  virtual bool quit();
  virtual bool read(yarp::os::ConnectionReader& connection) override;
  virtual std::vector<std::string> help(const std::string& functionName="--all");
};

#endif
