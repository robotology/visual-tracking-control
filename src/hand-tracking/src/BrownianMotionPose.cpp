/*
 * Copyright (C) 2016-2019 Istituto Italiano di Tecnologia (IIT)
 *
 * This software may be modified and distributed under the terms of the
 * BSD 3-Clause license. See the accompanying LICENSE file for details.
 */

#include <BrownianMotionPose.h>

#include <cmath>
#include <iostream>
#include <utility>

#include <BayesFilters/directional_statistics.h>

using namespace bfl;
using namespace Eigen;


BrownianMotionPose::BrownianMotionPose(const double q_x, const double q_y, const double q_z, const double q_yaw, const double q_pitch, const double q_roll, const unsigned int seed) noexcept :
    q_x_(q_x),
    q_y_(q_y),
    q_z_(q_z),
    q_yaw_(q_yaw),
    q_pitch_(q_pitch),
    q_roll_(q_roll),
    generator_(std::mt19937_64(seed)),
    distribution_pos_x_(std::normal_distribution<double>(0.0, q_x_)),
    distribution_pos_y_(std::normal_distribution<double>(0.0, q_y_)),
    distribution_pos_z_(std::normal_distribution<double>(0.0, q_z_)),
    distribution_yaw_(std::normal_distribution<double>(0.0, q_yaw_)),
    distribution_pitch_(std::normal_distribution<double>(0.0, q_pitch_)),
    distribution_roll_(std::normal_distribution<double>(0.0, q_roll_)),
    gaussian_random_pos_x_([&] { return (distribution_pos_x_)(generator_); }),
    gaussian_random_pos_y_([&] { return (distribution_pos_y_)(generator_); }),
    gaussian_random_pos_z_([&] { return (distribution_pos_z_)(generator_); }),
    gaussian_random_yaw_([&] { return (distribution_yaw_)(generator_); }),
    gaussian_random_pitch_([&] { return (distribution_pitch_)(generator_); }),
    gaussian_random_roll_([&] { return (distribution_roll_)(generator_); })
{ }


BrownianMotionPose::BrownianMotionPose(const double q_x, const double q_y, const double q_z, const double q_yaw, const double q_pitch, const double q_roll) noexcept :
    BrownianMotionPose(q_x, q_y, q_z, q_yaw, q_pitch, q_roll, 1)
{ }


BrownianMotionPose::BrownianMotionPose() noexcept :
    BrownianMotionPose(0.005, 0.005, 0.005, 0.1, 0.1, 0.1, 1)
{ }


BrownianMotionPose::BrownianMotionPose(const BrownianMotionPose& brown) :
    q_x_(brown.q_x_),
    q_y_(brown.q_y_),
    q_z_(brown.q_z_),
    q_yaw_(brown.q_yaw_),
    q_pitch_(brown.q_pitch_),
    q_roll_(brown.q_roll_),
    generator_(brown.generator_),
    distribution_pos_x_(brown.distribution_pos_x_),
    distribution_pos_y_(brown.distribution_pos_y_),
    distribution_pos_z_(brown.distribution_pos_z_),
    distribution_yaw_(brown.distribution_yaw_),
    distribution_pitch_(brown.distribution_pitch_),
    distribution_roll_(brown.distribution_roll_),
    gaussian_random_pos_x_(brown.gaussian_random_pos_x_),
    gaussian_random_pos_y_(brown.gaussian_random_pos_y_),
    gaussian_random_pos_z_(brown.gaussian_random_pos_z_),
    gaussian_random_yaw_(brown.gaussian_random_yaw_),
    gaussian_random_pitch_(brown.gaussian_random_pitch_),
    gaussian_random_roll_(brown.gaussian_random_roll_) { }

BrownianMotionPose::BrownianMotionPose(BrownianMotionPose&& brown) noexcept :
    q_x_(brown.q_x_),
    q_y_(brown.q_y_),
    q_z_(brown.q_z_),
    q_yaw_(brown.q_yaw_),
    q_pitch_(brown.q_pitch_),
    q_roll_(brown.q_roll_),
    generator_(std::move(brown.generator_)),
    distribution_pos_x_(std::move(brown.distribution_pos_x_)),
    distribution_pos_y_(std::move(brown.distribution_pos_y_)),
    distribution_pos_z_(std::move(brown.distribution_pos_z_)),
    distribution_yaw_(std::move(brown.distribution_yaw_)),
    distribution_pitch_(std::move(brown.distribution_pitch_)),
    distribution_roll_(std::move(brown.distribution_roll_)),
    gaussian_random_pos_x_(std::move(brown.gaussian_random_pos_x_)),
    gaussian_random_pos_y_(std::move(brown.gaussian_random_pos_y_)),
    gaussian_random_pos_z_(std::move(brown.gaussian_random_pos_z_)),
    gaussian_random_yaw_(std::move(brown.gaussian_random_yaw_)),
    gaussian_random_pitch_(std::move(brown.gaussian_random_pitch_)),
    gaussian_random_roll_(std::move(brown.gaussian_random_roll_))
{
    brown.q_x_        = 0.0;
    brown.q_y_        = 0.0;
    brown.q_z_        = 0.0;
    brown.q_yaw_      = 0.0;
    brown.q_pitch_    = 0.0;
    brown.q_roll_     = 0.0;
}


BrownianMotionPose::~BrownianMotionPose() noexcept { }


BrownianMotionPose& BrownianMotionPose::operator=(const BrownianMotionPose& brown)
{
    BrownianMotionPose tmp(brown);
    *this = std::move(tmp);

    return *this;
}


BrownianMotionPose& BrownianMotionPose::operator=(BrownianMotionPose&& brown) noexcept
{
    q_x_     = brown.q_x_;
    q_y_     = brown.q_y_;
    q_z_     = brown.q_z_;
    q_yaw_   = brown.q_yaw_;
    q_pitch_ = brown.q_pitch_;
    q_roll_  = brown.q_roll_;

    generator_             = std::move(brown.generator_);
    distribution_pos_x_    = std::move(brown.distribution_pos_x_);
    distribution_pos_y_    = std::move(brown.distribution_pos_y_);
    distribution_pos_z_    = std::move(brown.distribution_pos_z_);
    distribution_yaw_      = std::move(brown.distribution_yaw_);
    distribution_pitch_    = std::move(brown.distribution_pitch_);
    distribution_roll_     = std::move(brown.distribution_roll_);
    gaussian_random_pos_x_ = std::move(brown.gaussian_random_pos_x_);
    gaussian_random_pos_y_ = std::move(brown.gaussian_random_pos_y_);
    gaussian_random_pos_z_ = std::move(brown.gaussian_random_pos_z_);
    gaussian_random_yaw_   = std::move(brown.gaussian_random_yaw_);
    gaussian_random_pitch_ = std::move(brown.gaussian_random_pitch_);
    gaussian_random_roll_  = std::move(brown.gaussian_random_roll_);

    brown.q_x_     = 0.0;
    brown.q_y_     = 0.0;
    brown.q_z_     = 0.0;
    brown.q_yaw_   = 0.0;
    brown.q_pitch_ = 0.0;
    brown.q_roll_  = 0.0;

    return *this;
}


void BrownianMotionPose::propagate(const Eigen::Ref<const Eigen::MatrixXd>& cur_state, Eigen::Ref<Eigen::MatrixXd> prop_state)
{
    prop_state = cur_state;
}


void BrownianMotionPose::motion(const Eigen::Ref<const Eigen::MatrixXd>& cur_state, Eigen::Ref<Eigen::MatrixXd> mot_state)
{
    propagate(cur_state, mot_state);

    MatrixXd sample(6, mot_state.cols());
    sample = getNoiseSample(mot_state.cols());

    mot_state.topRows<3>() += sample.topRows<3>();
    mot_state.bottomRows<3>() = directional_statistics::directional_add(mot_state.bottomRows<3>(), sample.bottomRows<3>());
}


Eigen::MatrixXd BrownianMotionPose::getNoiseSample(const std::size_t num)
{
    MatrixXd sample(6, num);

    for (std::size_t i = 0; i < num; ++i)
    {
        sample(0, i) = gaussian_random_pos_x_();
        sample(1, i) = gaussian_random_pos_y_();
        sample(2, i) = gaussian_random_pos_z_();
        sample(3, i) = gaussian_random_yaw_();
        sample(4, i) = gaussian_random_pitch_();
        sample(5, i) = gaussian_random_roll_();
    }

    return sample;
}


std::pair<std::size_t, std::size_t> BrownianMotionPose::getOutputSize() const
{
    return std::make_pair(3, 3);
}
