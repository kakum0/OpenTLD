#include<tld_utils.h>
#include <opencv2/opencv.hpp>

//每个金字塔层的搜索窗口尺寸
class LKTracker{
private:
  std::vector<cv::Point2f> pointsFB;
  cv::Size window_size;//每个金字塔层的搜索窗口尺寸
  int level; //最大的金字塔层数
  std::vector<uchar> status;//数组。如果对应特征的光流被发现，数组中的每一个元素都被设置为 1， 否则设置为 0
  std::vector<uchar> FB_status;
  std::vector<float> similarity;//相似度
  std::vector<float> FB_error; //Forward-Backward error方法，求FB_error的结果与原始位置的欧式距离做比较，把距离过大的跟踪结果舍弃
  float simmed;
  float fbmed;
  //TermCriteria模板类，取代了之前的CvTermCriteria，这个类是作为迭代算法的终止条件的
  //该类变量需要3个参数，一个是类型，第二个参数为迭代的最大次数，最后一个是特定的阈值。
  //指定在每个金字塔层，为某点寻找光流的迭代过程的终止条件。
  cv::TermCriteria term_criteria;
  float lambda;
  void normCrossCorrelation(const cv::Mat& img1,const cv::Mat& img2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
  bool filterPts(std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2);
public:
  LKTracker();
  bool trackf2f(const cv::Mat& img1, const cv::Mat& img2,
                std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);
  float getFB(){return fbmed;}
};

