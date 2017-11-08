#include <DrawFwdKinPoses.h>

#include <utility>

using namespace bfl;
using namespace Eigen;


DrawFwdKinPoses::DrawFwdKinPoses() noexcept { }


DrawFwdKinPoses::~DrawFwdKinPoses() noexcept { }


DrawFwdKinPoses::DrawFwdKinPoses(DrawFwdKinPoses&& pf_prediction) noexcept :
    state_model_(std::move(pf_prediction.state_model_)),
    exogenous_model_(std::move(pf_prediction.exogenous_model_)) { };


DrawFwdKinPoses& DrawFwdKinPoses::operator=(DrawFwdKinPoses&& pf_prediction) noexcept
{
    state_model_ = std::move(pf_prediction.state_model_);

    exogenous_model_ = std::move(pf_prediction.exogenous_model_);

    return *this;
}


StateModel& DrawFwdKinPoses::getStateModel()
{
    return *state_model_;
}


void DrawFwdKinPoses::setStateModel(std::unique_ptr<StateModel> state_model)
{
    state_model_ = std::move(state_model);
}


ExogenousModel& DrawFwdKinPoses::getExogenousModel()
{
    return *exogenous_model_;
}


void DrawFwdKinPoses::setExogenousModel(std::unique_ptr<ExogenousModel> exogenous_model)
{
    exogenous_model_ = std::move(exogenous_model);
}


void DrawFwdKinPoses::predictStep(const Ref<const MatrixXf>& prev_states, const Ref<const VectorXf>& prev_weights,
                                              Ref<MatrixXf> pred_states, Ref<VectorXf> pred_weights)
{
    VectorXf sorted_cor = prev_weights;
    std::sort(sorted_cor.data(), sorted_cor.data() + sorted_cor.size());
    float threshold = sorted_cor.tail(6)(0);

    exogenous_model_->setProperty("ICFW_DELTA");

    for (int j = 0; j < prev_weights.rows(); ++j)
    {
        VectorXf tmp_states = VectorXf::Zero(prev_states.rows());

        if (!getSkipExogenous())
            exogenous_model_->propagate(prev_states.col(j), tmp_states);
        else
            tmp_states = prev_states.col(j);

        if (!getSkipState() && prev_weights(j) <= threshold)
            state_model_->motion(tmp_states, pred_states.col(j));
        else
            pred_states.col(j) = tmp_states;
    }

    pred_weights = prev_weights;
}
