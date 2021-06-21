
#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/core.hpp>
#include "dataStructures.h"

//======LERNER'S MODIFICATION STARTS
//(Code for Node and KdTree taken from src/quiz/cluster/kdtree.h)
//An additional struct clust implemented with functions in //src/quiz/cluster/cluster.cpp===========
#include <unordered_set>
#include<math.h>


// Structure to represent node of kd tree
struct Node
{
	std::vector<double> point;
	int id;
	Node* left;
	Node* right;

	Node(std::vector<double> arr, int setId)
	:	point(arr), id(setId), left(NULL), right(NULL)
	{}
    ~Node()
    {
    	delete left;
        delete right;
                
    }
};

struct KdTree
{
	Node* root;

	KdTree()
	: root(NULL)
	{}
    ~KdTree()
    {
    	delete root;
    }
	void insertHelper(Node** node,uint depth,std::vector<double>point,int id)
	{
		//Tree is empty
		if(*node == NULL)
			*node = new Node(point,id);
		else
		{
			//Calculate current dimension
			//uint cd = depth % 2;
				uint cd = depth % 3;
			if(point[cd]<((*node)->point[cd]))
				insertHelper(&((*node)->left),depth+1,point,id);
			else
				insertHelper(&((*node)->right),depth+1,point,id);
		}
	}
	void insert(std::vector<double> point, int id)
	{
		// TODO: Fill in this function to insert a new point into the tree
		// the function should create a new node and place correctly with in the root 
		insertHelper(&root,0,point,id);
	}

	void searchHelper(std::vector<double>target,Node* node,int depth,float distanceTol,std::vector<int> &ids)
	{
		if(node!=NULL)
		{
			if((node->point[0]>=(target[0]-distanceTol)&&node->point[0]<=(target[0]+distanceTol))&&(node->point[1]>=(target[1]-distanceTol)&&node->point[1]<=(target[1]+distanceTol))&&(node->point[2]>=(target[2]-distanceTol)&&node->point[2]<=(target[2]+distanceTol)))
			{
				float distance = sqrt((node->point[0]-target[0])*(node->point[0]-target[0])+(node->point[1]-target[1])*(node->point[1]-target[1])+(node->point[2]-target[2])*(node->point[2]-target[2]));
				if(distance<=distanceTol)
					ids.push_back(node->id);
			}

			//Check across boundary
			if((target[depth%3]-distanceTol)<node->point[depth%3])
				searchHelper(target,node->left,depth+1,distanceTol,ids);
			if((target[depth%3]+distanceTol)>node->point[depth%3])
				searchHelper(target,node->right,depth+1,distanceTol,ids);
		}
	}
	// return a list of point ids in the tree that are within distance of target
	std::vector<int> search(std::vector<double> target, float distanceTol)
	{
		std::vector<int> ids;
		searchHelper(target,root,0,distanceTol,ids);
		return ids;
	}
	

};
//Learner implemented struct with functions from
//src/quiz/cluster/cluster.cpp===========
struct Clust
{
  
void clusterHelper(int indice,const std::vector<std::vector<double>>points,std::vector<int>& cluster,std::vector<bool> &processed,KdTree* tree,float distanceTol)
{
	processed[indice] = true;
	cluster.push_back(indice);
	std::vector<int> nearest = tree->search(points[indice],distanceTol);

	for(int id:nearest)
	{
		if(!processed[id])
		clusterHelper(id,points,cluster,processed,tree,distanceTol);
	}
}
std::vector<std::vector<int>> euclideanCluster(const std::vector<std::vector<double>>& points, KdTree* tree, float distanceTol)
{

	// return list of indices for each cluster

	std::vector<std::vector<int>> clusters;
	std::vector<bool> processed(points.size(),false);

	int i =0;
	while(i<points.size())
	{
		if(processed[i])
		{
			i++;
			continue;
		}
		std::vector<int> cluster;
		clusterHelper(i,points,cluster,processed,tree,distanceTol);
		clusters.push_back(cluster);
		i++;
	}
 
	return clusters;
}


};
//========LERNER'S MODIFICATION ENDS=========


void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT);
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches);
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame);

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait=true);

void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg=nullptr);
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC);                  
#endif /* camFusion_hpp */
