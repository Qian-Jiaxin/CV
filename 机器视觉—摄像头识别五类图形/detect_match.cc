/*-----------------------------------*/
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include <vector>
#include <math.h>
/*-----------------------------------*/
//change in the future
//cv::Mat Circle_frame;
cv::Mat Mask;
//cv::Mat Mask= cv::Mat::zeros(500, 500, CV_8UC3);
/*-----------------------------------*/
//bool TestReadData(cv::String filename , std::vector<cv::Point2i> &ptdata);
void Detect_Shape(std::vector<cv::Point2i> pt);
bool video_record(std::vector<cv::Point2i> &ptdata);
bool pretreat_record(cv::Mat src , std::vector<cv::Point2i> &ptdata);
int D_Line(std::vector<cv::Point2i> angle);
void D_Triangle(std::vector<cv::Point2i> angle);
void D_Rectangle(std::vector<cv::Point2i> angle);
void D_Round(std::vector<cv::Point2i> angle);
/*-----------------------------------*/
//open camera and record the shape
bool video_record(std::vector<cv::Point2i> &ptdata)
{
	using namespace cv;
	using namespace std;
	//VideoCapture video(0);
	VideoCapture video("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)360,format=(string)I420, framerate=(fraction)25/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");
	Mat Circle_frame,frame;
	Mat roi,maskroi;
	Mat hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
	int hsize = 16;
	int calc_flag = 0;
	int point_flag = 1;
	int slection_flag =0;
	int vmin = 10, vmax = 256, smin = 30;
	float hranges[] = {0,180};
	const float* phranges = hranges;
	vector<Vec3f> circles;
	Point start,end;
	Rect trackWindow,selection;
	vector<Mat> maskroi_channels;
	
	video.read(Circle_frame);
	Mat Back_Ground= Mat::zeros(Circle_frame.size(),Circle_frame.type());
	Mask = cv::Mat::zeros(Circle_frame.size(),Circle_frame.type());

	while(true)
	{
		video.read(Circle_frame);
//		imshow("src",Circle_frame);
		
		if (waitKey(5) == 'q')
			break;
		Circle_frame.copyTo(frame);
		cvtColor(Circle_frame,hsv,COLOR_BGR2HSV);
		
		int _vmin = vmin, _vmax = vmax;
		inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),Scalar(180, 256, MAX(_vmin, _vmax)), mask);
		int ch[] = {0, 0};
		hue.create(hsv.size(), hsv.depth());
		mixChannels(&hsv, 1, &hue, 1, ch, 1);
	
		//calc_flag = 0时，进行霍夫圆检测
		if(calc_flag == 0)
		{
			imshow("src",Circle_frame);
			cvtColor(frame,frame,COLOR_BGR2GRAY);
			GaussianBlur(frame,frame,Size(5,5),2,2);
			HoughCircles(frame,circles,HOUGH_GRADIENT,1,frame.rows/4,150,40);
			if(circles.size() != 0)
			{	
				int cir_x = circles[0][0];
				int cir_y = circles[0][1];
				int r = circles[0][2];
					
				if((cir_x - r > 0) && (cir_y - r > 0) && (cir_x + r < Circle_frame.cols)&& (cir_y + r < Circle_frame.rows))
				{
					selection = Rect(cir_x - r,cir_y - r,2*r,2*r);
						
					maskroi = Circle_frame(Rect(cir_x - r,cir_y - r,2*r,2*r));
						
					imshow("roi",maskroi);
					
					if (waitKey(5) == 'r' && !maskroi.empty())
					{
						calc_flag = 1;
						frame.release();
						destroyWindow("roi");
					}
					Mat roi(hue, selection), maskroi(mask, selection);
					calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
					normalize(hist, hist, 0, 255, NORM_MINMAX);
					trackWindow = selection;
					histimg = Scalar::all(0);
					int binW = histimg.cols / hsize;
					Mat buf(1, hsize, CV_8UC3);
					for( int i = 0; i < hsize; i++ )
						buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
					cvtColor(buf, buf, COLOR_HSV2BGR);
					for( int i = 0; i < hsize; i++ )
					{
						int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
						rectangle( histimg, Point(i*binW,histimg.rows),Point((i+1)*binW,histimg.rows - val),Scalar(buf.at<Vec3b>(i)), -1, 8 );
//						imshow("histimg",histimg);
				  }
					
				}	
			}
		}
		else if(calc_flag == 1)
		{
			calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
			backproj &= mask;
			RotatedRect trackBox = CamShift(backproj, trackWindow,TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));	
			cout<<trackBox.center<<endl;	
			
			if(point_flag == 1)
			{
				destroyWindow("src");
				start = trackBox.center;
				end = start;
				point_flag = 2;
			}
			if(point_flag > 1)
			{
				end = trackBox.center;
				line(Back_Ground,start,end,Scalar(255,255,255),3,LINE_AA,0);
				imshow("Back_Ground",Back_Ground); 
				start = end;
			}
			     
      if (waitKey(5) == 'a' && !Back_Ground.empty())
      {
 				calc_flag = 2;    
      }

		}
		else
			break;
	}
	
	Circle_frame.release();

	pretreat_record(Back_Ground , ptdata);

	Back_Ground.release();

	return true;
}

//pretreat
bool pretreat_record(cv::Mat src , std::vector<cv::Point2i> &ptdata)
{
	using namespace cv;
	using namespace std;

	Mat dst_t,dst;
	vector<vector<Point>> contour;
	vector<Vec4i> hie;

	cvtColor(src, src, COLOR_BGR2GRAY);

	pyrDown(src, dst_t, Size(src.cols/2, src.rows/2));
	pyrUp(dst_t, dst, src.size());

	blur(dst,dst,Size(15,15));

//	imshow("dst",dst);

	findContours(dst, contour, hie, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	ptdata = contour[0];
	contour.clear();
	hie.clear();
	return true;
}

////import the vector
//bool TestReadData(cv::String filename , std::vector<cv::Point2i> &ptdata)
//{

//  cv::FileStorage fs;

//  if (!fs.open(filename, cv::FileStorage::READ)) {
//    // error message
//    std::cout << "Cannot open file: " << filename << std::endl;
//    return false;
//  }

//  cv::FileNode data_node = fs["data"];
//  for (cv::FileNodeIterator it = data_node.begin(); it != data_node.end(); ) {
//    int a, b;
//    *it++ >> a;
//    *it++ >> b;
//    ptdata.push_back(cv::Point2i(a, b));
//  }

//  fs.release();

//  for (auto it = ptdata.begin(); it != ptdata.end(); ++it) {
//    std::cout << *it;
//  }
//  std::cout << std::endl;

//	return true;
//}

//judge the shapes
void Detect_Shape(std::vector<cv::Point2i> pt)
{
	using namespace std;
	using namespace cv;

	string Shape_Name;
	int flag_line; 
	double epsilon = 0.048 * arcLength(pt , true);

	vector<Point2i> angle;
	approxPolyDP(pt,angle,epsilon,true);

	switch((int)angle.size())
	{
		case 2:{
			flag_line = D_Line(angle);
			if(flag_line == 1)
				Shape_Name = "vertical line";
			else
				Shape_Name = "horizonal line";
			break;
		}
		case 3:{
			D_Triangle(angle);
			Shape_Name = "triangle";
			break;
		}
		case 4:{
			D_Rectangle(angle);
			Shape_Name = "rectangle";
			break;
		}		
		default:{
			D_Round(angle);
			Shape_Name = "round";
			break;
		}
	}

	cout<<"It is a "<< Shape_Name <<endl;
	return;
}

//detect line
int D_Line(std::vector<cv::Point2i> angle)
{
	using namespace std;
	using namespace cv;

	int flag_line;
	
	int dy = abs(angle[0].y - angle[1].y);
	int dx = abs(angle[0].x - angle[1].x);
	double tan = dy/dx;
	
	if(tan > 1)
	{
		flag_line = 1;
		int x = (angle[0].x + angle[1].x)/2;
		line(Mask,Point(x,angle[0].y),Point(x,angle[1].y),Scalar(0,255,0),3);		
	}
	else
	{
		flag_line = 2;
		int y = (angle[0].y + angle[1].y)/2;
		line(Mask,Point(angle[0].x,y),Point(angle[1].x,y),Scalar(0,255,0),3);		
	}
	
	imshow("Line",Mask);
	
	return flag_line;
}

//detect triangle
void D_Triangle(std::vector<cv::Point2i> angle)
{
	using namespace std;
	using namespace cv;
	
	polylines(Mask, angle, true, Scalar(0,255,0), 3, LINE_AA);

	imshow("Triangle",Mask);

	return;
}

//detect rectangle
void D_Rectangle(std::vector<cv::Point2i> angle)
{
	using namespace std;
	using namespace cv;

	Point2f vertices[4];
	RotatedRect box = minAreaRect(angle);
	box.points(vertices);
	
	for(int i = 0;i<4;i++)
	{
		line(Mask, vertices[i], vertices[(i+1)%4], Scalar(0,255,0), 3);
	}

	imshow("Rectangle",Mask);

	return;
}

//detect round
void D_Round(std::vector<cv::Point2i> angle)
{
	using namespace std;
	using namespace cv;

	Point2f circle_centers;
	float circle_radius;

	minEnclosingCircle(angle, circle_centers, circle_radius);
	circle(Mask,circle_centers,circle_radius,Scalar(0,255,0),3,LINE_AA,0);

	imshow("Round",Mask);

	return;
}

int main(void) 
{
	using namespace std;
	using namespace cv;
	
	vector<Point2i> angle;
	
	while(1)
	{
		angle.clear();
		
		video_record(angle);
	
		//start judging what shape it is
		Detect_Shape(angle);

		while(waitKey(1) != 'q');
		destroyAllWindows();
	}

	return 1;	
}
