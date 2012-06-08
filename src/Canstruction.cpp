#include "Canstruction.h"

Canstruction::Canstruction(
    const cv::Mat& target,
    const std::vector<cv::Mat>& labels,
    const cv::Size& n_cans,
    int window_width):
    
    target_(target),
    labels_(labels),
    can_image_(target.rows, target.cols, target.type()),
    n_cans_(n_cans),
    window_size_(window_width, labels[0].rows),
    cans_(n_cans.width) {
  // set cans_ vector to contain correct can coordiates
  for (int x = 0; x < cans_.size(); ++x) {
    cans_[x] = std::vector<Can>(n_cans.height);
    for (int y = 0; y < cans_[x].size(); ++y) {
      cans_[x][y].set_x(x);
      cans_[x][y].set_y(y);
    }
  }
  // resize the target image to fit the cans (will stretch the target)
  cv::resize(target_, target_, cv::Size(n_cans.width  * window_size_.width,
                                       n_cans.height * window_size_.height));
  Optimize();
}

cv::Mat Canstruction::GetSubimage(int l, int offset, const cv::Size& target_size) {
  std::map<std::pair<int, int>, cv::Mat>::iterator it;
  it = label_offsets_.find(std::pair<int, int>(l, offset));
  // if not found
  if (it == label_offsets_.end()) {
    const int n_levels_up = 2;
    cv::Mat hl = GetHalfLabel(labels_[l], offset, n_levels_up);
    //cv::imshow("or", hl);
    GetCylindricalView(&hl);
    //cv::imshow("nw", hl);
    //cv::waitKey();
    for (int l = 0; l < n_levels_up; ++l) {
      cv::pyrDown(hl, hl);
    }
    cv::resize(hl, hl, target_size);
    label_offsets_[std::pair<int, int>(l, offset)] = hl;
  }
  return label_offsets_[std::pair<int, int>(l, offset)];
}

cv::Mat Canstruction::GetHalfLabel(const cv::Mat& label, double degrees, int upsample_by) {
  cv::Size half_size(static_cast<int>(label.cols / 3.0f), label.rows);
  // upsample
  cv::Mat label_up(label);
  for (int up_levels = 0; up_levels < upsample_by; ++up_levels) {
    cv::pyrUp(label_up, label_up);
    half_size.width *= 2;
    half_size.height *= 2;
  }
  // get the correct subimage (half of the can label)
  cv::Rect roi(0, 0, half_size.width, half_size.height);
  int n_steps = half_size.width * 2;  // total number of available steps
                                    // (discrete rotations)
  double steps_per_deg = n_steps / 360.0;
  roi.x = degrees * steps_per_deg;
  return label_up(roi);
}

void Canstruction::GetCylindricalView(cv::Mat* half_label) {
  const int n = 9;
  // divide into N high and thin segments
  int seg_width = half_label->cols / n;
  std::vector<cv::Mat> segments;
  segments.reserve(n);
  int total_width(0);
  for (int s = 0; s < n; ++s) {
    cv::Rect os_rect(s*seg_width, 0, seg_width, half_label->rows);  // original segment roi
    segments.push_back((*half_label)(os_rect));
    int new_width = segments[s].cols*cos(PI*(s-n/2)/n);
    if (new_width < 1) { continue; }
    // shrink each segment horizontally according to index (shorten the ones on the edges)
    cv::resize(segments[s], segments[s], cv::Size(new_width, segments[s].rows));
    cv::Rect ns_rect(total_width, 0, new_width, segments[s].rows);  // where to paste the new segment
    // copy shrinked segments into the original
    segments[s].copyTo((*half_label)(ns_rect));
    total_width += new_width;
  }
  // crop
  *half_label = (*half_label)(cv::Rect(0, 0, total_width, half_label->rows));  
}

std::pair<int, double> Canstruction::FindBestOffset(const cv::Mat& window_target, int l) {
  // sanity checks
  assert(window_target.rows == labels_[l].rows);
  assert(window_target.cols <= labels_[l].cols);
  if (window_target.cols == labels_[l].cols) {
    return std::pair<int, double>(0, Distance(window_target, labels_[l]));
  }
  // initialize the ROI
  cv::Rect window_rect(0, 0, window_size_.width, window_size_.height);
  // initialize searching params
  double min_dist = DBL_MAX;
  int best_offset = -1;
  // look through all possible offsets
  for (int offset = 0; offset < 360; offset+=6) {
    window_rect.x = offset;
    // get the dfistance from the target to the current offset in label
    cv::Mat subimage = GetSubimage(l, offset, cv::Size(window_target.cols, window_target.rows));
    double dist = Distance(window_target, subimage);
    // if that's the best so far, save it
    if (dist < min_dist) {
      best_offset = offset;
      min_dist = dist;
    }
  }
  return std::pair<int, double>(best_offset, min_dist);
}

std::pair<int, int> Canstruction::FindBestLabelAndOffset(const cv::Mat& window, const std::vector<cv::Mat>& labels) {
  double min_dist = DBL_MAX;
  int best_label = -1;
  int best_offset = -1;
  for (int l = 0; l < labels.size(); ++l) {
    std::pair<int, double> l_dist = FindBestOffset(window, l);
    if (l_dist.second < min_dist) {
      best_label = l;
      best_offset = l_dist.first;
      min_dist = l_dist.second; 
    }
  }
  return std::pair<int, int>(best_label, best_offset);
}

void Canstruction::Optimize() {
  // iterate over all cans
  for (int x = 0; x < cans_.size(); ++x) {
    for (int y = 0; y < cans_[x].size(); ++y) {
      // retrieve the target window
      cv::Rect roi(cans_[x][y].x()*window_size_.width, cans_[x][y].y()*window_size_.height, window_size_.width, window_size_.height);
      cv::Mat window_target(target_(roi));
      // for each window find the best rotation of the can
      std::pair<int, int> label_offset = FindBestLabelAndOffset(window_target, labels_);
      // save as that subimage
      cv::Mat best_fit(GetSubimage(label_offset.first, label_offset.second, roi.size()));
      best_fit.copyTo(can_image_(roi));
      // save can parameters ?
    }
  }
}