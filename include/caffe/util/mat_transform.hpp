#ifndef CAFFE_UTIL_MAT_TRANSFORM_H_
#define CAFFE_UTIL_MAT_TRANSFORM_H_
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#endif
#include "caffe/caffe.hpp"
namespace caffe {

inline void affineTransform(const cv::Mat &transformMatrix,const std::vector<cv::Point2f> &inputs, std::vector<cv::Point2f> &output)
{
    output.clear();
    for(auto point:inputs)
    {
        float x_new=transformMatrix.at<double>(0,0)*point.x+transformMatrix.at<double>(0,1)*point.y+transformMatrix.at<double>(0,2);
        float y_new=transformMatrix.at<double>(1,0)*point.x+transformMatrix.at<double>(1,1)*point.y+transformMatrix.at<double>(1,2);
        output.push_back(cv::Point2f(x_new,y_new));
    }
}

inline void affineTransform(const cv::Mat &transformMatrix,const NormalizedBBox &input_bbox, int img_width,int img_height,NormalizedBBox &output_bbox)
{
    float tmpX[]={input_bbox.xmin()*img_width,input_bbox.xmax()*img_width};
    float tmpY[]={input_bbox.ymin()*img_height,input_bbox.ymax()*img_height};
    float xmin=img_width,ymin=img_height,xmax=0,ymax=0;
//        std::cout<<rotateMatrix.at<double>(0,0)<<"--"<<rotateMatrix.at<double>(0,1)<<std::endl;
    for(int i=0;i<2;i++)
    {
        for(int j=0;j<2;j++)
        {
            float x=tmpX[i];
            float y=tmpY[j];
            float x_new=transformMatrix.at<double>(0,0)*x+transformMatrix.at<double>(0,1)*y+transformMatrix.at<double>(0,2);
            float y_new=transformMatrix.at<double>(1,0)*x+transformMatrix.at<double>(1,1)*y+transformMatrix.at<double>(1,2);
//                std::cout<<x<<","<<y<<","<<","<<x_new<<","<<y_new<<std::endl;
            xmin=std::min(x_new,xmin);
            ymin=std::min(y_new,ymin);
            xmax=std::max(x_new,xmax);
            ymax=std::max(y_new,ymax);
        }
    }
    output_bbox=input_bbox;
    output_bbox.set_xmin(xmin/img_width);
    output_bbox.set_ymin(ymin/img_height);
    output_bbox.set_xmax(xmax/img_width);
    output_bbox.set_ymax(ymax/img_height);
}

inline void affineTransform(const cv::Mat &transformMatrix,const NormalizedKeyPointGroup &input_keypoint_group,int img_width,int img_height,
                     NormalizedKeyPointGroup &output_keypoint_group)
{
    std::vector<cv::Point2f> input_vector;
    for(int i=0;i<input_keypoint_group.keypoint_size();i++)
    {
        const NormalizedKeyPoint &keypoint=input_keypoint_group.keypoint(i);
        input_vector.push_back(cv::Point2f(keypoint.x()*img_width,keypoint.y()*img_height));
    }
    std::vector<cv::Point2f> output_vector;
    affineTransform(transformMatrix,input_vector,output_vector);
    for(auto point : output_vector)
    {
        NormalizedKeyPoint *keypoint=output_keypoint_group.add_keypoint();
        keypoint->set_x(point.x/img_width);
        keypoint->set_y(point.y/img_height);
    }

}

inline void perspectiveTransform(const cv::Mat &transformMatrix,const NormalizedKeyPointGroup &input_keypoint_group,int img_width,int img_height,
                                 NormalizedKeyPointGroup &output_keypoint_group)
{
    std::vector<cv::Point2f> input_vector;
    for(int i=0;i<input_keypoint_group.keypoint_size();i++)
    {
        const NormalizedKeyPoint &keypoint=input_keypoint_group.keypoint(i);
        input_vector.push_back(cv::Point2f(keypoint.x()*img_width,keypoint.y()*img_height));
    }
    std::vector<cv::Point2f> output_vector;
    cv::perspectiveTransform(input_vector,output_vector,transformMatrix);
    for(auto point : output_vector)
    {
        NormalizedKeyPoint *keypoint=output_keypoint_group.add_keypoint();
        keypoint->set_x(point.x/img_width);
        keypoint->set_y(point.y/img_height);
    }
}

inline void perspectiveTransform(const cv::Mat &transformMatrix,const NormalizedBBox &input_bbox, int img_width,int img_height,NormalizedBBox &output_bbox)
{
    cv::Point2f lefttop(input_bbox.xmin()*img_width,input_bbox.ymin()*img_height);
    cv::Point2f rightbottom(input_bbox.xmax()*img_width,input_bbox.ymax()*img_height);
    cv::Point2f leftbottom(input_bbox.xmin()*img_width,input_bbox.ymax()*img_height);
    cv::Point2f righttop(input_bbox.xmax()*img_width,input_bbox.ymin()*img_height);

    std::vector<cv::Point2f> tmpInputPoints;
    tmpInputPoints.push_back(lefttop);
    tmpInputPoints.push_back(rightbottom);
    tmpInputPoints.push_back(leftbottom);
    tmpInputPoints.push_back(righttop);
    std::vector<cv::Point2f> tmpOutputPoints;
    tmpOutputPoints.resize(4);
    cv::perspectiveTransform(tmpInputPoints,tmpOutputPoints,transformMatrix);

    float xmin=img_width,ymin=img_height,xmax=0,ymax=0;
    for(int i=0;i<4;i++)
    {
        const cv::Point2f &p=tmpOutputPoints[i];
        xmin=std::min(p.x,xmin);
        ymin=std::min(p.y,ymin);
        xmax=std::max(p.x,xmax);
        ymax=std::max(p.y,ymax);
    }
    output_bbox=input_bbox;
    output_bbox.set_xmin(std::max(0.0f,xmin/img_width));
    output_bbox.set_ymin(std::max(0.0f,ymin/img_height));
    output_bbox.set_xmax(std::min(1.f,xmax/img_width));
    output_bbox.set_ymax(std::min(1.f,ymax/img_height));
}


}

#endif
