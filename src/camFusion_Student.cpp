
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"


using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        //putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
      putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 1, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        //putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
      putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 1, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
	///*
  boundingBox.kptMatches.clear();
  boundingBox.keypoints.clear();
  std::vector<cv::DMatch> tempMatches;
  
  for(auto elem:kptMatches)
    {
      if(boundingBox.roi.contains(kptsCurr[elem.trainIdx].pt))
      {
        tempMatches.push_back(elem);
              
        }
    }

  
  //code to eliminate outlier matches
  double distCurrSum = 0;
  double distPrevSum = 0;
  std::vector<double> distCurrs;
  std::vector<double> distPrevs;
  double avgCurrs=0, avgPrevs=0;
  
  for (auto it1 = tempMatches.begin();it1 != tempMatches.end() - 1; ++it1)
	{ // outer kpt. loop
      // get current keypoint and its matched partner in the prev. frame
		 cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
		 cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
      
		for (auto it2 = tempMatches.begin() + 1; it2 != tempMatches.end(); ++it2)
		{ // inner kpt.-loop
          	
			cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
			cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);
			// compute distances and distance ratios
          
			distCurrSum += cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            
			distPrevSum += cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
            
        }//eof inner loops
        double distCurrAvg = distCurrSum/(tempMatches.size()-1);
        double distPrevAvg = distPrevSum/(tempMatches.size()-1);
        distCurrs.push_back(distCurrAvg);
        distPrevs.push_back(distPrevAvg);
        avgCurrs +=distCurrAvg;
        avgPrevs +=distPrevAvg;
        
        distCurrSum = 0;
        distPrevSum = 0;
  }//eof outer loop
  
  // adding average distances for final match
  cv::KeyPoint kpOuterCurr = kptsCurr.at((*(tempMatches.end()-1)).trainIdx);
	cv::KeyPoint kpOuterPrev = kptsPrev.at((*(tempMatches.end()-1)).queryIdx);
 
  for (auto it1 = tempMatches.begin();it1 != tempMatches.end()- 1; ++it1)
  {
  	
    cv::KeyPoint kpInnerCurr = kptsCurr.at(it1->trainIdx);
	cv::KeyPoint kpInnerPrev = kptsPrev.at(it1->queryIdx);
    distCurrSum += cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
    distPrevSum += cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
   
  }//eof loop for final match
  
  double distCurrAvg = distCurrSum/(tempMatches.size()-1);
  double distPrevAvg = distPrevSum/(tempMatches.size()-1);
  distCurrs.push_back(distCurrAvg);//vector with average distances from each current keypoint 
  
  	distPrevs.push_back(distPrevAvg);//vector with average distances from each previous frame keypoint 
  avgCurrs +=distCurrAvg;//average distance of all current frame points
  avgCurrs = avgCurrs/tempMatches.size();
  avgPrevs +=distPrevAvg;//average distance of all previous frame points
  avgPrevs = avgPrevs/tempMatches.size();
  
 for(auto it=tempMatches.begin();it!=tempMatches.end();++it)
 {
 	double elemC = distCurrs[(int)(it-tempMatches.begin())];
    double elemP = distPrevs[(int)(it-tempMatches.begin())];
	 	if(((elemP>avgPrevs/10)||(elemP<5*avgPrevs))&&((elemC>avgCurrs/10)||(elemC<5*avgCurrs)))
        {
        	boundingBox.kptMatches.push_back(*it);
            boundingBox.keypoints.push_back(kptsCurr.at(it->trainIdx));
        }
 }
  
  
}



// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,                      std::vector<cv::DMatch>kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
  // compute distance ratios between all matched keypoints
	vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
  
	for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
	{ // outer kpt. loop
      // get current keypoint and its matched partner in the prev. frame
		 cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
		 cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
      
		for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
		{ // inner kpt.-loop
          	double minDist = 100.0; // min. required distance
			// get next keypoint and its matched partner in the prev. frame
			cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
			cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);
			// compute distances and distance ratios
			double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
			double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
			if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
			{ // avoid division by zero
				double distRatio = distCurr / distPrev;
				distRatios.push_back(distRatio);
			}
		} // eof inner loop over all matched kpts
      	} // eof outer loop over all matched kpts
  	// only continue if list of distance ratios is not empty
	if (distRatios.size() == 0)
	{
		TTC = NAN;
		return;
	}
	// compute camera-based TTC from distance ratios

	
    sort(distRatios.begin(),distRatios.end());
	double medianDistRatio;
  	int medIndice;
	if(distRatios.size()%2==0)
      medIndice = distRatios.size()/2;
	else
      medIndice = (distRatios.size()-1)/2;
	medianDistRatio=distRatios.at(medIndice);
	double dT = 1 / frameRate;
	TTC = -dT/(1-medianDistRatio);
    
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
 	double dT = 1/frameRate;// time between two measurements in second
  	double laneWidth =4.0;
	KdTree* tree = new KdTree;
    Clust* clust = new Clust; 
    std::vector<std::vector<double>>pointsPrev;//pointcloud points(x,y,z) from previous frame
    float distanceTolerance = 0.5;//max distance between cluster elements
    int iPrev = 0;
    int clustPrevSize=0;
    int clustPrevIndice;//cluster indice for largest cluster in previous frame bounding box


    // code for robust finding of minimum distance from previous frame

    for (auto it = lidarPointsPrev.begin();it != lidarPointsPrev.end();++it) 
      {   
        pointsPrev.push_back({(*it).x,(*it).y,(*it).z});
        tree->insert(pointsPrev[iPrev],iPrev); 
        if(it == lidarPointsPrev.end()-1)
          continue;
        ++iPrev;
      }  
    std::vector<std::vector<int>> clustPrevIndices =  clust->euclideanCluster(pointsPrev,tree,distanceTolerance);
    for(auto it=clustPrevIndices.begin();it!=clustPrevIndices.end();++it)
    {
      if((*it).size()>clustPrevSize)
      {
        clustPrevSize = (*it).size();
        clustPrevIndice =(int)(it-clustPrevIndices.begin());
      }
    }
    delete tree;
  
  std::cout<<"Prev loop ended"<<std::endl;
    //find closest distance to lidar point
    double minXPrev = 1e9, minXCurr = 1e9;
    for(auto it=clustPrevIndices[clustPrevIndice].begin();it!=clustPrevIndices[clustPrevIndice].end();++it)
    {//if(abs((lidarPointsPrev.begin()+(*it))->y)<=laneWidth/2.0)
       minXPrev = minXPrev>(lidarPointsPrev.begin()+(*it))->x?(lidarPointsPrev.begin()+(*it))->x:minXPrev;
    }
    std::cout<<"Closest Prev dist found"<<std::endl;
    
    // code for robust finding of minimum distance from current frame
    KdTree* tree1 = new KdTree;
   
    std::vector<std::vector<double>> pointsCurr;// pointcloud points(x,y,z) from current frame 
    int iCurr = 0;
    int clustCurrSize=0;
    int clustCurrIndice;//cluster indice for largest cluster in previous frame bounding box
    for (auto it = lidarPointsCurr.begin();it != lidarPointsCurr.end();++it) 
      {   
        pointsCurr.push_back({(*it).x,(*it).y,(*it).z});
        tree1->insert(pointsCurr[iCurr],iCurr); 
        if(it == lidarPointsCurr.end()-1)
          continue;
        ++iCurr;
      }  
    std::vector<std::vector<int>> clustCurrIndices =  clust->euclideanCluster(pointsCurr,tree,distanceTolerance);
    for(auto it=clustCurrIndices.begin();it!=clustCurrIndices.end();++it)
    {
      if((*it).size()>clustCurrSize)
      {
        clustCurrSize = (*it).size();
        clustCurrIndice =(int)(it-clustCurrIndices.begin());
      }
    }
  	delete tree1;
  	
    std::cout<<"Curr loop ended"<<std::endl;
    //find closest distance to lidar point
    for(auto it=clustCurrIndices[clustCurrIndice].begin();it!=clustCurrIndices[clustCurrIndice].end();++it)
    {//if(abs((lidarPointsCurr.begin()+(*it))->y)<=laneWidth/2.0)
       minXCurr = minXCurr>(lidarPointsCurr.begin()+(*it))->x?(lidarPointsCurr.begin()+(*it))->x:minXCurr;
    }

    std::cout<<"Curr min dist found"<<std::endl;

    //compute TTC from both measurements
    TTC = minXCurr*dT/(minXPrev - minXCurr);
  std::cout<<"TTC FOUND= "<<TTC<<std::endl;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
  
std::vector<std::vector<int>> tmps;
for(auto it=(prevFrame.boundingBoxes).begin();it!=(prevFrame.boundingBoxes).end();++it)
  {
    int boxIDprev =it->boxID;
  
    int temp = 0;
    
    for(auto it1=(currFrame.boundingBoxes).begin();it1!=(currFrame.boundingBoxes).end();++it1)
    {
      int temp1 = 0;
      int boxIDcurr1 = it1->boxID;
      std::vector<int> tmp;
      bool crossChk = false;
      for(auto it2=matches.begin();it2!=matches.end();++it2)
        {
          if((it1->roi.contains(currFrame.keypoints[it2->trainIdx].pt))&&(it->roi.contains(prevFrame.keypoints[it2->queryIdx].pt)))
       
            ++temp1;
        }//eof loop it2
      
      if(temp1>temp)
      {
      	temp = temp1;
        tmp = {boxIDprev,boxIDcurr1,temp};
        for(auto i=tmps.begin();i!=tmps.end();++i)
        {
          if((*i).at(1)==boxIDcurr1&&(*i).at(2)>temp)
          {
              crossChk = true;
              continue;
          }
          if((*i).at(1)==boxIDcurr1)
          {
            tmps.erase(i);
            --i;
          }
          
        }//eof loop i
        if(crossChk == false)
            tmps.push_back(tmp);
        crossChk = false;
      }
    }//eof loop it1
  }//eof loop it

for(auto i=tmps.begin();i!=tmps.end();++i)
{ 
  bbBestMatches.insert(std::make_pair((*i).at(0),(*i).at(1)));
}

}
