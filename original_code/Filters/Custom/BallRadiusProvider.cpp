#include "Filters/Custom/BallRadiusProvider.hpp"

#include "rhoban_utils/timing/benchmark.h"
#include "rhoban_utils/util.h"

#include "CameraState/CameraState.hpp"

using rhoban_utils::Benchmark;

#define USE_TBB 1

namespace Vision
{
namespace Filters
{

#if USE_TBB
//Taken from https://stackoverflow.com/questions/54466572/how-to-properly-multithread-in-opencv-in-2019
class Parallel_process : public cv::ParallelLoopBody
{

private:
    Utils::CameraState *cs;
    cv::Mat& m_img;
    int col;
    int max_row;
    double step;

public:
    Parallel_process(cv::Mat& outImage, Utils::CameraState *csVal, int colVal, double stepVal, int max_row_val)
        : cs(csVal), m_img(outImage), col(colVal), step(stepVal), max_row(max_row_val)
    {
    }

    virtual void operator()(const cv::Range& range) const
    {
        for(int row_n = range.start; row_n < range.end; row_n++)
        {
          int row = row_n * step;
          if(row>max_row) row = max_row;
            /* divide image in 'diff' number of parts and process simultaneously */
            /*
            cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i, img.cols, img.rows/diff));
            cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, retVal.cols, retVal.rows/diff));

            cv::medianBlur(in, out, size);
            */
            double ballRadius;
            ballRadius = cs->computeBallRadiusFromPixel(cv::Point2f(col, row), Vision::Utils::CAMERA_WIDE_FULL);
            // For points above horizon or outside camera FOV, set ballRadius to 0
            if (ballRadius < 0) ballRadius = 0;
            //ret.at<float>(row, col) = ballRadius;
            m_img.ptr<float>(row)[col] = ballRadius;         
        }
    }
};
#endif

std::string BallRadiusProvider::getClassName() const
{
  return "BallRadiusProvider";
}

int BallRadiusProvider::expectedDependencies() const
{
  return 1;
}

void BallRadiusProvider::setParameters()
{
  nbCols = ParamInt(4, 2, 200);
  nbRows = ParamInt(4, 2, 200);
  params()->define<ParamInt>("nbCols", &nbCols);
  params()->define<ParamInt>("nbRows", &nbRows);
}

void BallRadiusProvider::process()
{
  cv::Size size = getDependency().getImg()->size();
  // 1: Compute key columns and key rows
  std::vector<int> key_cols, key_rows;
  // 1.a: always select extreme pixels
  key_cols.push_back(0);
  key_rows.push_back(0);
  // 1.b: add intermediary values
  double step_x = size.width / (double)nbCols;
  double step_y = size.height / (double)nbRows;
  for (int col = 1; col < nbCols - 1; col++)
  {
    key_cols.push_back((int)(col * step_x));
  }
  for (int row = 1; row < nbRows - 1; row++)
  {
    key_rows.push_back((int)(row * step_y));
  }
  // 1.c: always use last pixel
  key_cols.push_back(size.width - 1);
  key_rows.push_back(size.height - 1);

  // 2: create image
  cv::Mat tmp_img(size, CV_32FC1);

  // 3: Place values at key points
  Benchmark::open("keypoints calculation");  
  for (int col : key_cols)
  {
    #if USE_TBB
    cv::parallel_for_(cv::Range(0, nbRows+1), Parallel_process(tmp_img, &getCS(), col, step_y, size.height - 1)); //nbRows+1 here to surely process last pixel
    #else
    for (int row : key_rows)
    {
      double ballRadius;
      if(size.width==1440) { //[Sol] Dirty hack for test only
        ballRadius = getCS().computeBallRadiusFromPixel(cv::Point2f(col, row), Vision::Utils::CAMERA_WIDE_FULL);
      } else {
        ballRadius = getCS().computeBallRadiusFromPixel(cv::Point2f(col, row));
      }
      // For points above horizon, set ballRadius to 0
      if (ballRadius < 0)
        ballRadius = 0;
      tmp_img.at<float>(row, col) = ballRadius;
    }
    #endif
  }
  Benchmark::close("keypoints calculation");

  
  /*for (int col : key_cols)
  {
    for (int row : key_rows)
    {
      double ballRadius = tmp_img.at<float>(row, col);
      std::cout << (int)ballRadius << " ";
    }
    std::cout << std::endl;
  }*/

  

  // 4: Interpolate on key columns
  // Note: forbidding interpolation when there are 0 values because their meaning is also that we failed to compute ball
  // radius, therefore it might provide degenerate result
  Benchmark::open("interpolation");
  for (int col : key_cols)
  {
    for (int row_idx = 0; row_idx < nbRows - 1; row_idx++)
    {
      int start_row = key_rows[row_idx];
      int end_row = key_rows[row_idx + 1];
      double start_val = tmp_img.at<float>(start_row, col);
      double end_val = tmp_img.at<float>(end_row, col);
      int dist = end_row - start_row;
      double diff = end_val - start_val;      
      double slope = diff / dist;
      bool interpolation_allowed = start_val > 0 && end_val > 0;
      for (int row = start_row + 1; row < end_row; row++)
      {
        double val = 0;
        if (interpolation_allowed)
        {
          val = start_val + slope * (row - start_row);
        }
        tmp_img.at<float>(row, col) = val;
      }
    }
  }

  // 5: Interpolate between key rows
  // Note: forbidding interpolation when there are 0 values because their meaning is also that we failed to compute ball
  // radius, therefore it might provide degenerate result
  for (int row = 0; row < size.height; row++)
  {
    for (int col_idx = 0; col_idx < nbCols - 1; col_idx++)
    {
      int start_col = key_cols[col_idx];
      int end_col = key_cols[col_idx + 1];
      int dist = end_col - start_col;
      double start_val = tmp_img.at<float>(row, start_col);
      double end_val = tmp_img.at<float>(row, end_col);
      double diff = end_val - start_val;
      double slope = diff / dist;
      bool interpolation_allowed = start_val > 0 && end_val > 0;
      for (int col = start_col + 1; col < end_col; col++)
      {
        double val = 0;
        if (interpolation_allowed)
        {
          val = start_val + slope * (col - start_col);
        }
        tmp_img.at<float>(row, col) = val;
      }
    }
  }
  Benchmark::close("interpolation");
  
  
  // 6: Affect img
  img() = tmp_img;
}
}  // namespace Filters
}  // namespace Vision
