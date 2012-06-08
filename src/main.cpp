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
  std::string target_path("C:\\Work\\VS2010\\canstruction\\img\\queen\\queen15.jpg");
  cv::Mat target(cv::imread(target_path));
  // get file names
  std::vector<std::string> file_paths;
  file_paths.push_back("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can1.jpg");
  file_paths.push_back("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can2.jpg");
  file_paths.push_back("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can3.jpg");
  file_paths.push_back("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can4.jpg");
  file_paths.push_back("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can5.jpg");
  file_paths.push_back("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can6.jpg");
  file_paths.push_back("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can7.jpg");
  file_paths.push_back("C:\\Work\\VS2010\\canstruction\\img\\heinz\\Can8.jpg");
  // read in the files and store them in a vector (as well as their rotations)
  std::vector<cv::Mat> labels;
  cv::Mat temp_label;
  for (int fp = 0; fp < file_paths.size(); ++fp) {
    temp_label = cv::imread(file_paths[fp]);
    labels.push_back(temp_label);
    labels.push_back(Canstruction::Rotate(temp_label));
  }

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