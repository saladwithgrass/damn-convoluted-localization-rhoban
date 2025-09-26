//Taken from https://github.com/errollw/EyeTab by errollw

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <tbb/tbb.h>

#include <iostream>
#include <stdio.h>
#include <time.h>

#include "ConicSection.h"
#include "fit_ellipse_utils.h"

using namespace std;

int random(int min, int max)
{
	static bool init = false;

	if (!init) {
		srand(time(NULL));
		init = true;
	}

	return rand()%(max-min)+min; 
}

// adapted from https://bitbucket.org/Leszek/pupil-tracker/ - Lech Swirski
// robustly fit ellipse to potential limbus points with RANSAC
cv::RotatedRect fit_ellipse(std::vector<cv::Point2f> edgePoints, cv::Mat iresx, cv::Mat iresy){

    // Number of points needed for a model
    const int n = 5;

    cv::RotatedRect elPupil;

    if (edgePoints.size() >= n) // Minimum points for ellipse
    {
        // Number of ransac iterations
        int k = 100;

        // Use TBB for RANSAC
        struct EllipseRansac_out {
            std::vector<cv::Point2f> bestInliers;
            cv::RotatedRect bestEllipse;
            double bestEllipseGoodness;
            int earlyRejections;
            bool earlyTermination;

            EllipseRansac_out() : bestEllipseGoodness(-std::numeric_limits<double>::infinity()), earlyTermination(false), earlyRejections(0) {}
        };

        struct EllipseRansac {
            const std::vector<cv::Point2f>& edgePoints;
            int n;
            const cv::Rect& bb;
            const cv::Mat& mDX;
            const cv::Mat& mDY;
            int earlyRejections;
            bool earlyTermination;

            EllipseRansac_out out;

            EllipseRansac(
                const std::vector<cv::Point2f>& edgePoints,
                int n,
                const cv::Rect& bb,
                const cv::Mat& mDX,
                const cv::Mat& mDY)
                : edgePoints(edgePoints), n(n), bb(bb), mDX(mDX), mDY(mDY), earlyTermination(false), earlyRejections(0)
            {
            }

            EllipseRansac(EllipseRansac& other, tbb::split)
                : edgePoints(other.edgePoints), n(other.n), bb(other.bb), mDX(other.mDX), mDY(other.mDY), earlyTermination(other.earlyTermination), earlyRejections(other.earlyRejections)
            {
                //std::cout << "Ransac split" << std::endl;
            }

            void operator()(const tbb::blocked_range<size_t>& r)
            {
                if (out.earlyTermination)
                        return;
                //std::cout << "Ransac start (" << (r.end()-r.begin()) << " elements)" << std::endl;
                for( size_t i=r.begin(); i!=r.end(); ++i )
                {
                    // Ransac Iteration
                    // ----------------
                    std::vector<cv::Point2f> sample;
                    sample = randomSubset(edgePoints, n);

                    cv::RotatedRect ellipseSampleFit = fitEllipse(sample);

                    // Normalise ellipse to have width as the major axis.
                    if (ellipseSampleFit.size.height > ellipseSampleFit.size.width)
                    {
                        ellipseSampleFit.angle = std::fmod(ellipseSampleFit.angle + 90, 180);
                        std::swap(ellipseSampleFit.size.height, ellipseSampleFit.size.width);
                    }

                    // Discard useless ellipses early
                    const int Radius_Max = 400;
                    const int Radius_Min = 20;
					cv::Size s = ellipseSampleFit.size;
					cv::Rect rb = ellipseSampleFit.boundingRect();

                    if (!ellipseSampleFit.center.inside(boundingBox(mDX))						// ellipse center outside eye-ROI center
						|| rb != (boundingBox(mDX) & rb)							// ellipse bounds outside eye-ROI
						|| s.height > Radius_Max*2								// other shape constraints ...
						|| s.width > Radius_Max*2
						|| s.height < Radius_Min 
                        || s.width < Radius_Min
						|| s.height > 4*s.width
						|| s.width > 4*s.height 
                        || s.width < 1
                        || s.height < 1
                        ) {
                        continue;
                    }

                    // Use conic section's algebraic distance as an error measure
                    ConicSection conicSampleFit(ellipseSampleFit);

                    // Check if sample's gradients are correctly oriented
                    /*bool gradientCorrect = true;
                    for(const cv::Point2f& p : sample) {
						cv::Point2f grad = conicSampleFit.algebraicGradientDir(p);
						float dx = mDX(cv::Point(p.x, p.y));
						float dy = mDY(cv::Point(p.x, p.y));

						float dotProd = dx*grad.x + dy*grad.y;

						gradientCorrect &= dotProd > 0;
                    }

					// If gradients not matched, EARLY REJECTION
                    if (!gradientCorrect)
                        continue;
                    */
                    // Assume that the sample is the only inliers

                    cv::RotatedRect ellipseInlierFit = ellipseSampleFit;
                    ConicSection conicInlierFit = conicSampleFit;
                    std::vector<cv::Point2f> inliers, prevInliers;

                    // Iteratively find inliers, and re-fit the ellipse
                    const int InlierIterations = 3;
                    for (int i = 0; i < InlierIterations; ++i)
                    {
                        // Get error scale for 1px out on the minor axis
                        cv::Point2f minorAxis(-std::sin(CV_PI/180.0*ellipseInlierFit.angle), std::cos(CV_PI/180.0*ellipseInlierFit.angle));
                        cv::Point2f minorAxisPlus1px = ellipseInlierFit.center + (ellipseInlierFit.size.height/2 + 1)*minorAxis;
                        float errOf1px = conicInlierFit.distance(minorAxisPlus1px);
                        float errorScale = 1.0f/errOf1px;

                        // Find inliers
                        inliers.reserve(edgePoints.size());
                        const float MAX_ERR = 5; //2;
                        for(const cv::Point2f& p : edgePoints) {
                            //float err = errorScale*conicInlierFit.distance(p);
                            float err = fabs(conicInlierFit.algebraicDistance(p));

                            if (err*err < MAX_ERR*MAX_ERR)
                                inliers.push_back(p);
                        }

                        if (inliers.size() < n) {
                            inliers.clear();
                            continue;
                        }

                        // Refit ellipse to inliers
                        ellipseInlierFit = fitEllipse(inliers);

                        if((ellipseInlierFit.size.width < Radius_Min) || (ellipseInlierFit.size.height < Radius_Min) ) {
                            inliers.clear();
                            continue;                            
                        }

                        conicInlierFit = ConicSection(ellipseInlierFit);

                        // Normalise ellipse to have width as the major axis.
                        if (ellipseInlierFit.size.height > ellipseInlierFit.size.width){
                            ellipseInlierFit.angle = std::fmod(ellipseInlierFit.angle + 90, 180);
                            std::swap(ellipseInlierFit.size.height, ellipseInlierFit.size.width);
                        }
                    }
                    if (inliers.empty()
						//||!ellipseSampleFit.center.inside(bb)						// ellipse center outside eye-ROI center
						//|| r != (boundingBox(mDX) & r)
                        )
                        continue;

                    // Calculate ellipse goodness (Image aware support)
                    double ellipseGoodness = 0;
                    for(cv::Point2f& p : inliers) {
                        //cv::Point2f grad = conicInlierFit.algebraicGradientDir(p);
                        //float dx = mDX(p);
                        //float dy = mDY(p);
                        //double edgeStrength = dx*grad.x + dy*grad.y;
                        if((p.x>0)&&(p.x<mDX.cols)&&(p.y>0)&&(p.y<mDX.rows) ) {
                            //unsigned char dx = mDX.at<unsigned char>(p.y, p.x);
                            //unsigned char dy = mDY.at<unsigned char>(p.y, p.x);
                            //double edgeStrength = dx + dy;
                            double edgeStrength = fabs(conicInlierFit.algebraicDistance(p));
                            ellipseGoodness += edgeStrength;
                        }
                    }
                    //ellipseGoodness = ellipseGoodness / (ellipseInlierFit.size.width/2.0 + ellipseInlierFit.size.height/2.0);

					// If ellipse goodness is high enough, store new ellipse (no early termination)
                    if (ellipseGoodness > out.bestEllipseGoodness) {
                        std::swap(out.bestEllipseGoodness, ellipseGoodness);
                        std::swap(out.bestInliers, inliers);
                        std::swap(out.bestEllipse, ellipseInlierFit);
                    }
                }
            }

            void join(EllipseRansac& other) {

				// RANSAC join
                if (other.out.bestEllipseGoodness > out.bestEllipseGoodness) {
                    std::swap(out.bestEllipseGoodness, other.out.bestEllipseGoodness);
                    std::swap(out.bestInliers, other.out.bestInliers);
                    std::swap(out.bestEllipse, other.out.bestEllipse);
                }
                out.earlyRejections += other.out.earlyRejections;
                earlyTermination |= other.earlyTermination;

                out.earlyTermination = earlyTermination;
            }
        };

		// iris center must be close to refined ROI center
		//cv::Rect bbPupil = roiAround(mPupilSobelX.size().width/2, mPupilSobelX.size().width/2, int(mPupilSobelX.size().width*0.5));

        EllipseRansac ransac(edgePoints, n, cv::Rect(), iresx, iresy);

		try {
            tbb::parallel_reduce(tbb::blocked_range<size_t>(0,k,k/8), ransac);
        }
        catch (std::exception& e) {
            const char* c = e.what();
            std::cerr << e.what() << std::endl;
        }

        cv::RotatedRect ellipseBestFit = ransac.out.bestEllipse;
        elPupil = ellipseBestFit;
    }

    return elPupil;
}