#include "Canstruction.h"

// transform label for optimization:
// 1 - resize to fit within the specified window
// 2 - repeat at edges to allow circular overflow
void TransformLabel(cv::Mat* label, const cv::Size& window_size) {
  cv::resize(*label, *label, window_size);
  
  cv::Rect label_whole(0, 0, label->cols, label->rows); // ROI for the whole label
  cv::Rect label_left(0, 0, label->cols/2, label->rows);  // ROI for the left half of the label
  cv::Rect temp_left(0, 0, label->cols, label->rows); // ROI fo the left part of temp
  cv::Rect temp_right(label->cols, 0, label->cols/2, label->rows);  // ROI fo the right part of temp

  cv::Mat temp(label->rows, static_cast<int>(label->cols * 1.5f), label->type());
  // copy whole label into the beginning of temp
  label->copyTo(temp(temp_left));
  // copy the left half of label into the end of temp
  (*label)(label_left).copyTo(temp(temp_right));
  *label = temp;
}

int main(int argc, char *argv[]) {
  cv::Mat target(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\queen\\queen15.jpg"));
  std::vector<cv::Mat> labels;
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can1.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can2.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can3.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can4.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can5.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can6.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can7.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can8.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can1_r.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can2_r.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can3_r.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can4_r.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can5_r.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can6_r.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can7_r.jpg"));
  labels.push_back(cv::imread("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can8_r.jpg"));

  int window_width(20);
  cv::Size n_cans(26, 43);

  for (int l = 0; l < labels.size(); ++l) {
    cv::Size window_size(0, window_width);
    // DBG ratio changed by a factor of 2 to imitate half-cans
    double ratio = static_cast<double>(labels[l].cols) / static_cast<double>(labels[l].rows/2);
    window_size.width = window_size.height * ratio;
    TransformLabel(&labels[l], window_size);
  }

  Canstruction cs(target, labels, n_cans, window_width);
  cv::Mat result = cs.can_image();
  cv::imshow("w", result);
  cv::waitKey();
}