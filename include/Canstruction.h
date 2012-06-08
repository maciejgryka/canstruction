#ifndef CANSTRUCTION_H
#define CANSTRUCTION_H

#include <vector>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define PI 3.141592653589793

class Can {
public:
  Can():
      x_(-1),
      y_(-1),
      rotation_(-1) {};
  Can(int x, int y, int rotation):
      x_(x),
      y_(y),
      rotation_(rotation) {};
  ~Can() {};

  int x() const { return x_; };
  int y() const { return y_; };
  int rotation() const { return rotation_; };

  void set_x(int x) { x_ = x; };
  void set_y(int y) { y_ = y; };
  void set_rotation(int rotation) { rotation_ = rotation; };
private:
  int x_;
  int y_;
  int rotation_;
};

class Canstruction {
public:
  Canstruction(
      const cv::Mat& target,
      const std::vector<cv::Mat>& labels,
      const cv::Size& n_cans,
      int window_width);
  ~Canstruction() {};

  cv::Mat can_image() const { return can_image_; };

  static cv::Mat Rotate(const cv::Mat& image) {
    cv::Point image_center(image.cols/2, image.rows/2);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(image_center, 180.0, 1.0);
    cv::Mat image_rot;
    cv::warpAffine(image, image_rot, rotation_matrix, cv::Size(image.cols, image.rows));
    return image_rot;
  };

private:
  double Distance(const cv::Mat& im1, const cv::Mat& im2) const {
    return cv::norm(im1, im2);
  };
  // returns a can view at the specified offset (rotation)
  cv::Mat GetSubimage(int l, int offset, const cv::Size& target_size);
  // return half of the label that will be visible at the specified rotation,
  // optionally upsampled by the specified amount
  // assumes that label that's passed in is actually 1.5 label
  cv::Mat GetHalfLabel(const cv::Mat& label, double degrees, int upsample_by = 1);
  // warp the half-label so that it looks like it's a texture applied to a cylinder
  void GetCylindricalView(cv::Mat* half_label);
  // returns the best offset and associated error
  std::pair<int, double> FindBestOffset(const cv::Mat& window, int l);
  // returns a vector that holds label index and offset of the best match
  std::pair<int, int> FindBestLabelAndOffset(const cv::Mat& window, const std::vector<cv::Mat>& labels);
  void Optimize();

  cv::Mat target_;
  std::vector<cv::Mat> labels_;
  cv::Mat can_image_;
  cv::Size n_cans_;
  cv::Size window_size_;
  std::vector<std::vector<Can> > cans_;

  std::map<std::pair<int, int>, cv::Mat> label_offsets_;
};

#endif // CAN_OPTIMIZER_H