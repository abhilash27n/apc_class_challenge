/**
 * @file BAXTER_SIM_application.hpp
 *
 * @copyright Software License Agreement (BSD License)
 * Copyright (c) 2013, Rutgers the State University of New Jersey, New Brunswick 
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Authors: Andrew Dobson, Andrew Kimmel, Athanasios Krontiris, Zakary Littlefield, Kostas Bekris
 *
 * Email: pracsys@googlegroups.com
 */
#pragma once

#ifndef PRX_BAXTER_SIM_APPLICATION_HPP
#define PRX_BAXTER_SIM_APPLICATION_HPP

#include "prx/utilities/definitions/defs.hpp"
#include "prx/simulation/applications/empty_application.hpp"
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include "ros/ros.h"    

//Proj includes
#include <tf/transform_listener.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
  

namespace prx
{
    namespace packages
    {
        namespace manipulation
        {
            class baxter_sim_application_t : public prx::sim::empty_application_t
            {

              public:
                baxter_sim_application_t();
                virtual ~baxter_sim_application_t();
                virtual void init(const util::parameter_reader_t * const reader);
                virtual void received_plan_callback(const prx_simulation::plan_msg& msg);
                virtual void planning_ready_callback(const std_msgs::String& msg);
                //adding our own callback
                virtual void cam_image_callback(const sensor_msgs::ImageConstPtr& msg);
                virtual void randomize_positions();
                virtual void frame(const ros::TimerEvent& event);
              private:
                ros::NodeHandle node;
                ros::Publisher manipulation_request_pub;      
                ros::Subscriber received_plan_sub;      
                ros::Subscriber planning_ready_sub;
                //ros::Subscriber tf_sub;
                //ros::Subscriber cam_info_sub;
                tf::TransformListener listener;
                tf::StampedTransform transform;
                image_transport::Subscriber cam_image_sub;
                int counter;
                sim::state_t* simulator_state;

            };
        }

    }
}

#endif
