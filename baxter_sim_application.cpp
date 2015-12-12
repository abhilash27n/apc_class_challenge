/**
 * @file baxter_sim_application.cpp
 *
 * @copyright Software License Agreement (BSD License)
 * Copyright (c) 2013, Rutgers the State University of New Jersey, New Brunswick 
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Authors: Andrew Dobson, Andrew Kimmel, Athanasios Krontiris, Zakary Littlefield, Rahul Shome, Kostas Bekris
 *
 * Email: pracsys@googlegroups.com
 */


#include "simulation/applications/baxter_sim_application.hpp"
#include "prx/utilities/definitions/string_manip.hpp"
#include "prx_simulation/plan_msg.h"
#include "prx_simulation/control_msg.h"
#include <pluginlib/class_list_macros.h>

#include "prx/simulation/communication/sim_base_communication.hpp"
#include "prx/simulation/communication/planning_comm.hpp"
#include "prx/simulation/communication/simulation_comm.hpp"
#include "prx/simulation/communication/visualization_comm.hpp"
#include "simulation/simulators/manipulation_simulator.hpp"
#include "simulation/systems/plants/movable_body_plant.hpp"
 #include "prx/simulation/systems/plants/plant.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#define PI 3.14

#define MAX_MANIPULATIONS 10


using namespace cv;
using namespace std;

//OPENCV CODE ###############################################################################
Mat hsv_image;
Mat src_gray;
int thresh = 200;
int max_thresh = 255;
RNG rng(12345);

//convert centroid to camera frame
// we took camera parameters from topic camera_info
double prin_x = 346.177; 
double prin_y = 247.518;
double cc_x = 410.117;
double cc_y = 410.511;
double cam_x,cam_y;
double cam_z = 0;

//store colors in order
double red_obj[2][3];
double green_obj[2][3];
double blue_obj[2][3];
double yellow_obj[2][3];
double unsure_obj[8][4];

int red_idx=0;
int blue_idx=0;
int green_idx=0;
int yellow_idx=0;
int unsure_idx=0;

/// Function header
int findColor(vector<Point>, double *);

std::string to_string(int i)
{
    std::stringstream ss;
    ss << i;
    return ss.str();
}
std::string to_string(float i)
{
    std::stringstream ss;
    ss << i;
    return ss.str();
}


/** @function main */
int DetectObject(Mat src)
{

    red_idx = 0;
    green_idx = 0;
    blue_idx = 0;
    yellow_idx = 0;
    unsure_idx = 0;

    for(int i =0;i<2;i++)
    {  
      for(int j=0;j<3;j++)
      {
        red_obj[i][j]=0;
        green_obj[i][j]=0;
        yellow_obj[i][j]=0;
        blue_obj[i][j]=0;
        unsure_obj[i][j]=0;
      }
    }

   // Convert image to hsv
   cvtColor( src, hsv_image, COLOR_BGR2HSV);
   
   /// Convert image to gray and blur it
   cvtColor( src, src_gray, COLOR_BGR2GRAY );
   blur( src_gray, src_gray, Size(3,3) );

   Mat threshold_output;
   vector<vector<Point> > contours;
   vector<Vec4i> hierarchy;

   /// Detect edges using Threshold
   threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
  
   /// Find contours
   findContours( threshold_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
   //cout<<"CONTOURS "<<contours.size()<<endl;

   // remove small contours
   int contour_size_threshold = 30;
   for (vector<vector<Point> >::iterator it = contours.begin(); it!=contours.end(); )
   {
        //cout<<" CONTOUR SIZES "<<it->size()<<endl;
       if (it->size()<contour_size_threshold)
           it=contours.erase(it);
       else
           ++it;
   }  

   //Finding minEllipse
   vector<RotatedRect> minEllipse( contours.size() );
   /// Find the convex hull object for each contour
   vector<vector<Point> >hull( contours.size() );
   /// Draw contours + hull results
   Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
   //Finding momentss
   vector<Moments> mu(contours.size() );
   //  Get the mass centers:
   vector<Point2f> mc( contours.size() );
   //to find color for all pixel points
   vector<Point> pixelPoints;
   //No of objects and color of text to be printed on image
   int object = 0;
   Scalar colorOfText = Scalar(225,225,225);
   for( int i = 0; i < contours.size(); i++ )
   { 
      //minimum ellipse
      minEllipse[i] = fitEllipse( Mat(contours[i]) ); 

      //find convex hull for contour
      convexHull( Mat(contours[i]), hull[i], false );

      //find color
      pixelPoints.clear();
      Mat labels = Mat::zeros(hsv_image.size(), CV_8UC1); 
      drawContours( labels, hull, i, 255, 10, 8, vector<Vec4i>(), 0, Point() );
      for(int m = 0; m < labels.rows; m++)
      {
         for(int n = 0; n < labels.cols; n++)
         {
            if(labels.at<uchar>(m,n) != 0)
               pixelPoints.push_back(Point(m,n));
         }
      }

      double prob = 0;
      int contour_color = findColor(pixelPoints, &prob);
      cout<<"Color Found With Probability : "<<prob<<endl;

      String colorString;

      //set color
      Scalar color;
      if(prob > 0.85)
      {
          if(contour_color == 1){
             color = Scalar( 0, 0, 255 );
             colorString = "Red";
          } else if(contour_color == 2){
             color = Scalar( 255, 0, 0 );
             colorString = "Blue";
          } else if(contour_color == 3){
             color = Scalar( 0, 255, 0);
             colorString = "Green";
          } else if(contour_color == 4){
             color = Scalar( 10, 255, 255 );
             colorString = "Yellow";
          } else {
             color = Scalar( 155, 155, 155 );
             colorString = "Unable to recognize";
          }
      }
      else
      {
          color = Scalar( 155, 155, 155 );
          colorString = "Not Sure";
      }

      drawContours( drawing, hull, i, color, -1, 8, vector<Vec4i>(), 0, Point() );

      //Finding moments
      mu[i] = moments( contours[i], false );

      //Getting centroid
      mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
      drawing(cv::Rect(mc[i].x, mc[i].y, 5, 5)) = 255;

      //draw Orientation angle
      float angle=361;
      int sleeping_object_threshold = 30;
      if(abs(minEllipse[i].size.width - minEllipse[i].size.height) > sleeping_object_threshold)
      {  

         angle = minEllipse[i].angle;
         if (minEllipse[i].size.width < minEllipse[i].size.height) 
         {
             angle = 90 + angle;
         }
         
         int length = 150;
         Point P1(mc[i].x,mc[i].y);
         Point P2;

         P2.x =  (int)round(P1.x + length * cos(angle * CV_PI / 180.0));
         P2.y =  (int)round(P1.y + length * sin(angle * CV_PI / 180.0));

         line(drawing, P1, P2, (255,255,255), 5);
      }

      object++;
      //Printing details object no, orientation, centroid location, color 
      string objText = "Object: " + to_string(object);
      string colorText = "Color: " + colorString;
      string centroidText = "Centroid: (" + to_string(mc[i].x) + "," + to_string(mc[i].y) + ")"; 
      String orienText = "Orientation: " + to_string(angle);
      putText(drawing, objText, Point(mc[i].x+5, mc[i].y), FONT_HERSHEY_COMPLEX_SMALL, 0.5, colorOfText, 1, 8);
      putText(drawing, colorText, Point(mc[i].x+5, mc[i].y+14), FONT_HERSHEY_COMPLEX_SMALL, 0.5, colorOfText, 1, 8);
      putText(drawing, centroidText, Point(mc[i].x+5, mc[i].y+28), FONT_HERSHEY_COMPLEX_SMALL, 0.5, colorOfText, 1, 8);
      putText(drawing, orienText, Point(mc[i].x+5, mc[i].y+42), FONT_HERSHEY_COMPLEX_SMALL, 0.5, colorOfText, 1, 8);

      cam_x =  ((mc[i].y - prin_y)*cam_z)/cc_y;
      cam_y =  -((mc[i].x - prin_x)*cam_z)/cc_x;
      
      //Store separate color objects
      //cout<<red_idx<<" "<<blue_idx<<" "<<green_idx<<" "<<yellow_idx<<endl;
      if(contour_color == 1)
      {
         //red
         if(prob > 0.85)
         {
          red_obj[red_idx][0] = cam_x;
          red_obj[red_idx][1] = cam_y;
          red_obj[red_idx][2] = angle;
          red_idx++;
         }
         else
         {
           unsure_obj[unsure_idx][0] = cam_x;
           unsure_obj[unsure_idx][1] = cam_y;
           unsure_obj[unsure_idx][2] = angle;
           unsure_obj[unsure_idx][3] = contour_color;
           unsure_idx++;
         }
      }
      else if(contour_color == 2)
      {
         if(prob > 0.85)
         {
           //blue
           blue_obj[blue_idx][0] = cam_x;
           blue_obj[blue_idx][1] = cam_y;
           blue_obj[blue_idx][2] = angle;
           blue_idx++;
         }
         else
         {
           unsure_obj[unsure_idx][0] = cam_x;
           unsure_obj[unsure_idx][1] = cam_y;
           unsure_obj[unsure_idx][2] = angle;
           unsure_obj[unsure_idx][3] = contour_color;
           unsure_idx++;
         }
      }
      else if(contour_color == 3)
      {
         if(prob > 0.85)
         {
           //green
           green_obj[green_idx][0] = cam_x;
           green_obj[green_idx][1] = cam_y;
           green_obj[green_idx][2] = angle;
           green_idx++;
         }
         else
         {
           unsure_obj[unsure_idx][0] = cam_x;
           unsure_obj[unsure_idx][1] = cam_y;
           unsure_obj[unsure_idx][2] = angle;
           unsure_obj[unsure_idx][3] = contour_color;
           unsure_idx++;
         }
      }
      else if(contour_color == 4)
      {
         if(prob > 0.85)
         {
           //yellow
           yellow_obj[yellow_idx][0] = cam_x;
           yellow_obj[yellow_idx][1] = cam_y;
           yellow_obj[yellow_idx][2] = angle;
           yellow_idx++;
         }
         else
         {
           unsure_obj[unsure_idx][0] = cam_x;
           unsure_obj[unsure_idx][1] = cam_y;
           unsure_obj[unsure_idx][2] = angle;
           unsure_obj[unsure_idx][3] = contour_color;
           unsure_idx++;
         }
      }

   }

   string totalObjText = "TOTAL OBJECTS FOUND: " +to_string(object);
   putText(drawing, totalObjText, Point(20, 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, colorOfText, 1, 8);
   /// Show in a window
   imshow( "Hull demo", drawing );

   return(0);
}

int findColor(vector<Point> contour_points, double *prob)
{

   int red_count = 0;         //1 = red
   int blue_count = 0;        //2 = blue
   int green_count = 0;       //3 = green
   int yellow_count = 0;      //4 = yellow


   //find which count is higher
   for(int i = 0; i < contour_points.size();i++){
      //cout << "BEGINNING: " << hsv_image.rows << " " << hsv_image.cols << " " << contour_points[i].x << " " << contour_points[i].y << endl;
      Vec3b pixel = hsv_image.at<Vec3b>(contour_points[i].x,contour_points[i].y); 
      //cout << "FINDING COLOR:" << i << endl;
      if(((pixel[0] >= 0 && pixel[0] <= 10) || (pixel[0] >= 160 && pixel[0] <= 180)) && pixel[1] >= 60 && pixel[2] >=50)
         red_count++;
      else if(pixel[0] >= 106 && pixel[0] <= 126 && pixel[1] >= 60 && pixel[2] >=90)
         blue_count++;
      else if(pixel[0] >= 60 && pixel[0] <= 102 && pixel[1] >= 85 && pixel[2] >=40)
         green_count++;
      else if(pixel[0] >=10 && pixel[0] <= 50 && pixel[1] >= 60 && pixel[2] >=50)
         yellow_count++;
  }

  cout<<"FIND COLOR :: RED : "<<red_count<<" BLUE : "<<blue_count<<" GREEN : "<<green_count<<" YELLOW : "<<yellow_count<<endl;
   //return whichever is color is higher
   if(red_count >= blue_count && red_count >= green_count && red_count >= yellow_count)
   {
    *prob = (double)red_count/(double)(red_count+blue_count+green_count+yellow_count);
    return 1;
   }

   if(blue_count >= red_count && blue_count >= green_count && blue_count >= yellow_count)
   {
    *prob = (double)blue_count/(double)(red_count+blue_count+green_count+yellow_count);
    return 2;
   }

   if(green_count >= red_count && green_count >= blue_count && green_count >= yellow_count)
   {
    *prob = (double)green_count/(double)(red_count+blue_count+green_count+yellow_count);
    return 3;
   }

   if(yellow_count >= red_count && yellow_count >= blue_count && yellow_count >=green_count)
   {
    *prob = (double)yellow_count/(double)(red_count+blue_count+green_count+yellow_count);
    return 4;
   }

   //If undetermined
   return 0;

 }

//############################################################################################

PLUGINLIB_EXPORT_CLASS( prx::packages::manipulation::baxter_sim_application_t, prx::sim::application_t)

namespace prx
{
    using namespace util;
    using namespace sim::comm;
    namespace packages
    {        
        namespace manipulation
        {
            double eerot[9];
            double eetrans[3];
            double eerot_cam[9];
            double eerot_res[9];
            double qx,qy,qz,qw;
            double x1,y1;

            baxter_sim_application_t::baxter_sim_application_t() { }

            baxter_sim_application_t::~baxter_sim_application_t() { }

            void baxter_sim_application_t::init(const parameter_reader_t * const reader)
            {
                prx::sim::empty_application_t::init(reader);
                PRX_DEBUG_COLOR("Initialized", PRX_TEXT_RED);
                received_plan_sub = node.subscribe("/ready_to_plan", 1, &baxter_sim_application_t::planning_ready_callback, this);
                planning_ready_sub = node.subscribe("/planning/plans", 1, &baxter_sim_application_t::received_plan_callback, this);
                //GETTING T and R matrices for camera to base transformation
                
                for(int i = 0; i<1000; i++)
                {
                    
                    try
                    {
                        listener.lookupTransform("/base", "/left_hand_camera_axis",ros::Time(0),transform);

                         //PRX_DEBUG_COLOR(" transform x : "<<  transform.getOrigin().x(), PRX_TEXT_RED);
                         //PRX_DEBUG_COLOR(" transform y : "<<  transform.getOrigin().y(), PRX_TEXT_RED);
                         //PRX_DEBUG_COLOR(" transform z : "<<  transform.getOrigin().z(), PRX_TEXT_RED);

                        // PRX_DEBUG_COLOR(" orientation x : "<<  transform.getRotation().x(), PRX_TEXT_RED);
                        // PRX_DEBUG_COLOR(" orientation y : "<<  transform.getRotation().y(), PRX_TEXT_RED);
                        // PRX_DEBUG_COLOR(" orientation z : "<<  transform.getRotation().z(), PRX_TEXT_RED);
                        // PRX_DEBUG_COLOR(" orientation w : "<<  transform.getRotation().w(), PRX_TEXT_RED);

                        eetrans[0] = transform.getOrigin().x();
                        eetrans[1] = transform.getOrigin().y();
                        eetrans[2] = transform.getOrigin().z();
                        qx = transform.getRotation().x();
                        qy = transform.getRotation().y();
                        qz = transform.getRotation().z();
                        qw = transform.getRotation().w();
                        
                        /*qx = 0;
                        qy = 0;
                        qz = 0.707;
                        qw = 0.707;*/

                        eerot[0] = (1.0f - 2.0f * qy * qy - 2.0f * qz * qz);
                        eerot[1] = (2.0f * qx * qy - 2.0f * qz*qw);
                        eerot[2] = (2.0f * qx * qz + 2.0f * qy*qw);
                        eerot[3] = (2.0f * qx * qy + 2.0f * qz*qw);
                        eerot[4] = (1.0f - 2.0f * qx * qx - 2.0f * qz*qz);
                        eerot[5] = (2.0f * qy * qz - 2.0f * qx*qw);
                        eerot[6] = (2.0f * qx * qz - 2.0f * qy*qw);
                        eerot[7] = (2.0f * qy * qz + 2.0f * qx*qw);
                        eerot[8] = (1.0f - 2.0f * qx * qx - 2.0f * qy*qy);

                        cam_z = 0.13 + eetrans[2];
                    }
                    catch (tf::TransformException &ex)
                    {
                        ROS_ERROR("%s",ex.what());
                        ros::Duration(1.0).sleep();
                        continue;
                    }
                }

                //GETTING IMAGE FROM THE CAMERA
                cv::namedWindow("view", WINDOW_AUTOSIZE);
                cv::namedWindow( "Hull demo", WINDOW_AUTOSIZE );
                cv::startWindowThread();
                image_transport::ImageTransport it(node);
                cam_image_sub = it.subscribe("/cameras/left_hand_camera/image", 1, &baxter_sim_application_t::cam_image_callback, this);
                
                // tf_sub = node.subscribe("/tf",1,&baxter_sim_application_t::tf_callback, this);
                //new subsribers end

                manipulation_request_pub = node.advertise<std_msgs::String> ("/manipulation_request", 1);
                simulator_state = simulator->get_state_space()->alloc_point();
                PRX_ERROR_S("\n\n\n SIMULATOR AT START::: "<<simulator->get_state_space()->print_point(simulator_state,4)<<"\n\n\n");
                simulator_running = true;
                counter = 0;
            }

            // void baxter_sim_application_t::tf_callback(const )
            // {

            // }

            void baxter_sim_application_t::cam_image_callback(const sensor_msgs::ImageConstPtr& msg)
            {
                 try
                  {
                    Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
                    cv::imshow("view", image);
                    //cv::waitKey(0);
                    cout<<"CALLING OBJECT DETECTION"<<endl;
                    DetectObject(image);
                    cout<<"OBJECT DETECTED"<<endl;
                    randomize_positions();
                    cout<<"POSITION HAS BEEN SET"<<endl;
                    cout<<"####################################################"<<endl;
                    cv::waitKey(25);
                  }
                  catch (cv_bridge::Exception& e)
                  {
                    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
                  }
            }

            void baxter_sim_application_t::planning_ready_callback(const std_msgs::String& msg)
            {
                randomize_positions();
                std_msgs::String send_msg;
                std::stringstream ss;
                ss<<"start_manipulate";
                send_msg.data = ss.str();
                manipulation_request_pub.publish(send_msg);
            }

            void baxter_sim_application_t::received_plan_callback(const prx_simulation::plan_msg& msg)
            {
                double duration = 0;
                foreach(const prx_simulation::control_msg& control_msg, msg.plan)
                {
                    duration += control_msg.duration;
                }
                PRX_ERROR_S("\n\n\nSleeping for : \n\n\n"<<duration*1.5<<" seconds");
                sleep(duration*1.5);
                if(counter++>MAX_MANIPULATIONS)
                {
                    PRX_ERROR_S("End of experiment!");
                    return;
                }
                //randomize_positions();
                std_msgs::String send_msg;
                std::stringstream ss;
                ss<<"manipulate";
                send_msg.data = ss.str();
                manipulation_request_pub.publish(send_msg);
            }

            void baxter_sim_application_t::randomize_positions()
            {
                std::vector<movable_body_plant_t* > objects;
                int red=0,blue=0,green=0,yellow=0,unsure=0;
                manipulation_simulator_t* manip_sim = dynamic_cast<manipulation_simulator_t* >(simulator);
                
                manip_sim->get_movable_objects(objects);
                
                for(int i=0;i<objects.size();i++)
                {
                    double cam_frame_x,cam_frame_y,cam_frame_angle;
                    std::string object_color = objects[i]->get_object_color();

                    if(object_color == "red")
                    {
                        //red_idx is the number of red objects detected
                        if(red < red_idx)
                        {
                          cam_frame_x = red_obj[red][0];
                          cam_frame_y = red_obj[red][1];
                          cam_frame_angle = red_obj[red][2];
                          cout<<"RED object "<<red<<" angle is : "<<cam_frame_angle<<endl;
                          red++;
                        }
                        else
                        {
                          if(unsure < unsure_idx && (unsure_obj[unsure][3] == 1))
                          {
                              cam_frame_x = unsure_obj[unsure][0];
                              cam_frame_y = unsure_obj[unsure][1];
                              cam_frame_angle = unsure_obj[unsure][2];
                              cout<<"RED object "<<red<<" angle is : "<<cam_frame_angle<<endl;
                              unsure++;
                          }
                          else
                          {
                            cam_frame_x = -5;
                            cam_frame_y = -5;
                            cam_frame_angle = 0;
                          }
                        }
                    }
                    else if(object_color == "green")
                    {
                        if(green < green_idx)
                        {
                          cam_frame_x = green_obj[green][0];
                          cam_frame_y = green_obj[green][1];
                          cam_frame_angle = green_obj[green][2];
                          cout<<"GREEN object "<<green<<" angle is : "<<cam_frame_angle<<endl;
                          green++;
                        }
                        else
                        {
                          if(unsure < unsure_idx && (unsure_obj[unsure][3] == 3))
                          {
                              cam_frame_x = unsure_obj[unsure][0];
                              cam_frame_y = unsure_obj[unsure][1];
                              cam_frame_angle = unsure_obj[unsure][2];
                              cout<<"GREEN object "<<red<<" angle is : "<<cam_frame_angle<<endl;
                              unsure++;
                          }
                          else
                          {
                            cam_frame_x = -5;
                            cam_frame_y = -5;
                            cam_frame_angle = 0;
                          }
                        }
                    }
                    else if(object_color == "blue")
                    {
                        if(blue < blue_idx)
                        {
                          cam_frame_x = blue_obj[blue][0];
                          cam_frame_y = blue_obj[blue][1];
                          cam_frame_angle = blue_obj[blue][2];
                          cout<<"BLUE object "<<blue<<" angle is : "<<cam_frame_angle<<endl;
                          blue++;
                        }
                        else
                        {
                          if(unsure < unsure_idx && (unsure_obj[unsure][3] == 2))
                          {
                              cam_frame_x = unsure_obj[unsure][0];
                              cam_frame_y = unsure_obj[unsure][1];
                              cam_frame_angle = unsure_obj[unsure][2];
                              cout<<"BLUE object "<<red<<" angle is : "<<cam_frame_angle<<endl;
                              unsure++;
                          }
                          else
                          {
                            cam_frame_x = -5;
                            cam_frame_y = -5;
                            cam_frame_angle = 0;
                          }
                        }
                    }
                    else if(object_color == "yellow")
                    {
                        if(yellow < yellow_idx)
                        {
                          cam_frame_x = yellow_obj[yellow][0];
                          cam_frame_y = yellow_obj[yellow][1];
                          cam_frame_angle = yellow_obj[yellow][2];
                          cout<<"YELLOW object "<<yellow<<" angle is : "<<cam_frame_angle<<endl;
                          yellow++;
                        }
                        else
                        {
                          if(unsure < unsure_idx && (unsure_obj[unsure][3] == 4))
                          {
                              cam_frame_x = unsure_obj[unsure][0];
                              cam_frame_y = unsure_obj[unsure][1];
                              cam_frame_angle = unsure_obj[unsure][2];
                              cout<<"YELLOW object "<<red<<" angle is : "<<cam_frame_angle<<endl;
                              unsure++;
                          }
                          else
                          {
                            cam_frame_x = -5;
                            cam_frame_y = -5;
                            cam_frame_angle = 0;
                          }
                        }
                    }

                    //Transform centroid from camera frame to global frame
                    x1 = eerot[0]*cam_frame_x + eerot[1]*cam_frame_y + eerot[2]*cam_z + eetrans[0];
                    y1 = eerot[3]*cam_frame_x + eerot[4]*cam_frame_y + eerot[5]*cam_z + eetrans[1];

                    const space_t* object_space = objects[i]->get_state_space();
                    
                    space_point_t* target_object = object_space->alloc_point();
                    

                    target_object->memory[0]=x1;
                    target_object->memory[1]=y1;
                    target_object->memory[2]=0.87;

                    //Vertical Objects
                    if(cam_frame_angle == 361)
                    {
                        target_object->memory[3]=0;
                        target_object->memory[4]=0;
                        target_object->memory[5]=0;
                        target_object->memory[6]=1;
                    } 
                    else
                    {
                        cam_frame_angle = 360 - cam_frame_angle;

                        double cos_a_2 = cos(cam_frame_angle*PI/360);
                        double sin_a_2 = sin(cam_frame_angle*PI/360);

                        //Camera Frame Orientation
                        //90 degrees about x- axis and
                        //by cam_frame_angle about z-axis
                        double cx =  0.707*cos_a_2;
                        double cy = -0.707*sin_a_2;
                        double cz = -0.707*sin_a_2;
                        double cw =  0.707*cos_a_2;

                        //Global Frame Orientation
                        double tx= qx*cw + qw*cx + qy*cz - qz*cy;
                        double ty= qw*cy - qx*cz + qy*cw + qz*cx;
                        double tz= qw*cz + qx*cy - qy*cx + qz*cw;
                        double tw= qw*cw - qx*cx - qy*cy - qz*cz;

                        target_object->memory[3]= tx;
                        target_object->memory[4]= ty;
                        target_object->memory[5]= tz;
                        target_object->memory[6]= tw;
                    }
                    
                    object_space->copy_from_point(target_object);
                    //objects[i]->print_configuration();
                }

                manip_sim->get_state_space()->copy_to_point(simulator_state);
                simulator->push_state(simulator_state);//By default the objects return to their original position
                //PRX_ERROR_S("SIMULATOR STATE SPACE CHANGED TO::: "<<simulator->get_state_space()->print_memory(4));
            }

            void baxter_sim_application_t::frame(const ros::TimerEvent& event)
            {
                handle_key();

                if( simulator_mode == 1 )
                {
                    if( simulator_counter > 0 )
                    {
                        simulator_running = true;
                        simulator_counter--;
                        loop_timer.reset();
                    }
                    else
                        simulator_running = false;
                }
                if( loop_timer.measure() > 1.0 )
                    loop_timer.reset();
                if( simulator_running )
                {
                    if( replays_states )
                    {
                        if( loop_counter > (int)replay_simulation_states.size() )
                            loop_counter = 0;
                        simulation_state_space->copy_from_point(replay_simulation_states[loop_counter]);
                        simulation_state_space->copy_to_point(simulation_state);
                    }
                    else
                    {
                        simulator->push_control(simulation_control);
                        simulator->propagate_and_respond();
                    }
                    if( stores_states )
                        store_simulation_states.copy_onto_back(simulation_state);

                    if( screenshot_every_simstep )
                    {
                        ((visualization_comm_t*)vis_comm)->take_screenshot(0, 1);
                    }
                    loop_total += loop_timer.measure_reset();
                    loop_counter++;
                    loop_avg = loop_total / loop_counter;

                }

                if(visualization_counter++%visualization_multiple == 0)
                {
                    tf_broadcasting();
                }



            }
        }


    }
}
