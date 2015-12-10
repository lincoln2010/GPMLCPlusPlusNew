// ALR.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include <iostream>
#include<fstream>
#include <direct.h> 
#include <io.h> 
#include "GPML.h"
using namespace cv;
using namespace std;
void string_split(std::string& s, std::string& delim,std::vector< double > &ret)  
{  
    size_t last = 0;  
    size_t index=s.find_first_of(delim,last);  
    while (index!=std::string::npos)  
    {  
		string sf = s.substr(last,index-last);
		ret.push_back(atof(sf.c_str()));  
        last=index+1;  
        index=s.find_first_of(delim,last);  
    }  
    if (index-last>0)  
    {  
		string sf = s.substr(last,index-last);
		ret.push_back(atof(sf.c_str()));  
    }  
} 
Mat readFileToMat(string fileName)
{
    ifstream in(fileName);

	Mat m;

	string line;
	while(getline(in, line))
	{
		vector<double> ret;
		std::string s_ = ",";
		string_split(line, s_, ret);
		if (ret.size() > 0)
		{


			Mat t(1, ret.size(), CV_64F);
			for (unsigned int i = 0; i < ret.size(); i++)
			{
				t.at<double>(i) = ret.at(i);
			}

			if (m.empty())
				m = t;
			else
			{
				cv::vconcat(m, t, m);
			}

		}
	}
	in.close();
    return m;
}


bool getImageFileList(const char *srcDir, vector<string> &fileList)  
{  
    _finddata_t fileDir;  
    long lfDir;  
    char f[100] = {0};  
  
    sprintf(f, "%s\\*.bmp", srcDir);  
    if((lfDir = _findfirst(f, &fileDir)) != -1L)  
    {  
        do  
        {  
            fileList.push_back(fileDir.name);  
        }while(_findnext(lfDir, &fileDir) == 0 );  
        _findclose(lfDir);  
  
        return true;  
    }  
  
    return false;  
}


bool sort_string(std::string &s1, std::string &s2) 
{ 
	string f1 = s1.substr(s1.find_first_of("(") + 1, s1.find_last_of(")") - s1.find_first_of("(") - 1);

	string f2 = s2.substr(s2.find_first_of("(") + 1, s2.find_last_of(")") - s2.find_first_of("(") - 1);


	int i1 = atoi(f1.c_str());
	int i2 = atoi(f2.c_str());

	if (i1 < i2)
		return true;
	else
		return false;
	
}
int fun_size(Mat &m, int i)
{
	CV_Assert(i == 1 || i == 2);
	if (i == 1)
		return m.rows;
	if (i == 2)
		return m.cols;
	CV_Assert(0);
	return 0;
}
Mat fun_getfeature(Mat &E0, int Hw, int Vw)
{
	//Mat E = Mat(1, Hw * Vw, CV_64F);
	std::vector<double> tmp;
	int pos = 0;
	for (int i = 1; i <= Hw; i++)
	{
		for (int j = 1; j <= Vw; j++)
		{
			Mat temp = E0(cv::Range(cvRound(1.0*(i-1)*fun_size(E0,1)/Hw), cvRound(1.0*i*fun_size(E0,1)/Hw)), 
				cv::Range(cvRound(1.0*(j-1)*fun_size(E0,2)/Vw), cvRound(1.0*j*fun_size(E0,2)/Vw)));
			Mat test;
			temp.copyTo(test);
			temp.isContinuous();
			cv::Scalar s = sum(sum(temp));
			//E.at<double>(pos++) = s[0];
			tmp.push_back(s[0]);
		}
	}

	if (tmp.size() == 0)
		return Mat();


	Mat E = Mat(1, tmp.size(), CV_64F);
	for (int i = 0; i < tmp.size(); i++)
		E.at<double>(i) = tmp[i];

	cv::Scalar s = sum(E);

	E = E / s[0];
	return E;
}

void getInfoFeatureImage(vector<string> &fileList,string &floder,  Mat &information_feature, Mat &information_image, int Hw, int Vw)
{
	for (int i = 0; i < fileList.size(); i++)
	{
		string s = floder + fileList.at(i);
		Mat im = imread(s, 0);
		Mat feature = fun_getfeature(im, Hw, Vw);
		if (information_feature.empty())
			information_feature = feature;
		else
			cv::vconcat(information_feature, feature, information_feature);

		//matlib to opencv, need to t
		im = im.t();
		if (information_image.empty())
			information_image = im.reshape(0, 1);
		else
			cv::vconcat(information_image, im.reshape(0, 1), information_image);
	}
}


void saveMat(string fileName, Mat &mat)
{
	cv::FileStorage fs(fileName, FileStorage::WRITE);
	CV_Assert(fs.isOpened());
	fs << "mat" << mat;
	fs.release();
}

void readMat(string fileName, Mat &mat)
{
	cv::FileStorage fs(fileName, FileStorage::READ);
	CV_Assert(fs.isOpened());
	fs["mat"] >> mat;
	fs.release();
}



void updateALR(Mat &ALR_x, Mat &ALR_y, Mat &E_update, Mat &information_xy_train, int i, int Hw, int Vw)
{
	Mat rowi = information_xy_train.row(i);
	int size_reature = Hw * Vw;

	Mat E = rowi(cv::Range(0,1), cv::Range(2,size_reature + 2 - 1));
	Mat x = rowi(cv::Range(0,1), cv::Range(0,1));
	Mat y = rowi(cv::Range(0,1), cv::Range(1,2));

	if (ALR_x.empty())
		ALR_x = x;
	else
		cv::hconcat(ALR_x, x, ALR_x);

	if (ALR_y.empty())
		ALR_y = y;
	else
		cv::hconcat(ALR_y, y, ALR_y);

	if (E_update.empty())
		E_update = E;
	else
		cv::vconcat(E_update, E, E_update);
}


int _tmain0(int argc, _TCHAR* argv[])
{

	


	//information_xy=load('./Train/s01/annotations.txt');
	cv::Mat  information_xy = readFileToMat("E:\\tsinghua\\C++GP\\Train\\s01\\annotations.txt");

	/*Hw = 3;  % ALR 高度维数
	Vw = 5;  % ALR 宽度维数
	Hw_ALRadvanced = 15;  % ALRadvanced高度维数
	Vw_ALRadvanced = 30;  % ALRadvanced宽度维数
	size_feature=Hw*Vw;  % ALR feature大小
	size_image=Hw_ALRadvanced*Vw_ALRadvanced; % ALRadvanced image大小
	x_imagesize=36;  % 原始图片的大小
	y_imagesize=60;
	size_xyimage=x_imagesize*y_imagesize;
	*/

	long Hw = 3;
	long Vw = 5;
	long Hw_ALRadvanced = 15;
	long Vw_ALRadvanced = 30;
	long size_feature = Hw*Vw;

	long size_image = Hw_ALRadvanced * Vw_ALRadvanced;
	long x_imagesize = 36;
	long y_imagesize = 60;

	long size_xyimage = x_imagesize * y_imagesize;

	
	std::string floder = "C:\\Users\\TZ\\Desktop\\C++GP\\Train\\s01\\126";
	
	vector<string> sortfile;
	getImageFileList(floder.c_str(), sortfile);

	//sort input name
	std::sort(sortfile.begin(), sortfile.end(), sort_string);

	Mat information_feature, information_image;
	getInfoFeatureImage(sortfile, floder , information_feature, information_image, Hw, Vw);


	saveMat("information_feature.yml", information_feature);
	saveMat("information_image.yml", information_image);

	Mat m(information_xy, cv::Range::all(), cv::Range(0,2));
	cv::hconcat(m, information_feature, information_xy);



	int size_train = information_xy.rows;
	Mat position_Ori(information_xy, cv::Range::all(), cv::Range(0,2));

	Mat feature(information_xy, cv::Range::all(), cv::Range(2,2+size_feature));
	Mat image_Ori1 = information_image.clone();

	Mat image_Ori;
	for (int i = 0; i < size_train; i++)
	{
		Mat m = image_Ori1.row(i);
		m = m.reshape(0, y_imagesize);

		m = m.t();
		m = fun_getfeature(m, x_imagesize, y_imagesize);
		m *= size_xyimage;

		if (image_Ori.empty())
			image_Ori = m;
		else
			cv::vconcat(image_Ori, m, image_Ori);
	}

	saveMat("image_Ori.yml", image_Ori);

	Mat image;
	for (int i = 0; i < size_train; i++)
	{
		Mat m = image_Ori.row(i);
		m = m.reshape(0, y_imagesize);

		m = m.t();
		m = fun_getfeature(m, Hw_ALRadvanced, Vw_ALRadvanced);

		if (image.empty())
			image = m;
		else
			cv::vconcat(image, m, image);
	}

	saveMat("image.yml", image);

	Mat x;
	Mat y;
	normalize(position_Ori.col(0), x,-1.0,1.0,NORM_MINMAX);
	normalize(position_Ori.col(1), y,-1.0,1.0,NORM_MINMAX);


	Mat information_xy_train;
	for (int i = 0; i < size_train; i++)
	{
		Mat m = image_Ori.row(i);
		m = m.reshape(0, y_imagesize);
		m = m.t();
		m = fun_getfeature(m, Hw, Vw);
		
		if (information_xy_train.empty())
			information_xy_train = m;
		else
			cv::vconcat(information_xy_train,m, information_xy_train);
	}
	hconcat(y, information_xy_train, information_xy_train);
	hconcat(x, information_xy_train, information_xy_train);

	Mat information_image_train;
	for (int i = 0; i < size_train; i++)
	{
		Mat m = image_Ori.row(i);
		m = m.reshape(0, y_imagesize);
		m = m.t();
		m = fun_getfeature(m, Hw_ALRadvanced, Vw_ALRadvanced);
		
		if (information_image_train.empty())
			information_image_train = m;
		else
			cv::vconcat(information_image_train,m, information_image_train);
	}
	hconcat(y, information_image_train, information_image_train);
	hconcat(x, information_image_train, information_image_train);

	saveMat("information_image_train.yml", information_image_train);
	saveMat("information_xy_train.yml", information_xy_train);



	//update 阶段
	int i = 0;
	Mat ALR_x;
	Mat ALR_y;
	Mat E_update;
	for (i = 0; i < 150; i++)
	{
		updateALR(ALR_x, ALR_y, E_update, information_xy_train, i, Hw, Vw);
	}

	return 0;
}

int _tmain(int argc, _TCHAR* argv[])
{
	/*
	cv::Mat a;
	cv::FileStorage fs("E:\\tsinghua\\C++GP\\a.yml", cv::FileStorage::READ);
	fs["mat0000"] >> a;

	fs.release();

	GPML gpml;

	Mat c;// = gpml.sq_dist(a, Mat());

	a = a.t();
	c = gpml.covSEiso(HPY_conSEiso(0, 5.203782474663246), a);

	saveMat("C.yml", c);
	*/

	/*
	cv::Mat hyp, y, mu, s2;
	cv::FileStorage fs("C:\\Users\\TZ\\Desktop\\C++GP\\test1.yml", cv::FileStorage::READ);
	fs["mat0000"] >> hyp;
	
	fs.release();
	
	cv::FileStorage fs2("C:\\Users\\TZ\\Desktop\\C++GP\\test2.yml", cv::FileStorage::READ);
	fs2["mat0000"] >> y;
	fs2.release();
	cv::FileStorage fs3("C:\\Users\\TZ\\Desktop\\C++GP\\test3.yml", cv::FileStorage::READ);
	fs3["mat0000"] >> mu;
	fs3.release();
	cv::FileStorage fs4("C:\\Users\\TZ\\Desktop\\C++GP\\test4.yml", cv::FileStorage::READ);
	fs4["mat0000"] >> s2;
	fs4.release();

	GPML gpml;
	Ans_likGauss Ans;
	Ans = gpml.likGauss(hyp,y,mu,s2,3,Ans);
	*/

	/*
	cv::Mat hyp, x, m, A;
	cv::FileStorage fs1("C:\\Users\\TZ\\Desktop\\C++GP\\test_hyp4.yml", cv::FileStorage::READ);
	fs1["mat0000"] >> hyp;
	cv::FileStorage fs2("C:\\Users\\TZ\\Desktop\\C++GP\\test_x4.yml", cv::FileStorage::READ);
	fs2["mat0000"] >> x;
	cv::FileStorage fs3("C:\\Users\\TZ\\Desktop\\C++GP\\test_m4.yml", cv::FileStorage::READ);
	fs3["mat0000"] >> m;

	GPML gpml;
	A = gpml.meanConst(hyp,x);
	
	saveMat("A.yml", A);
	saveMat("hyp.yml", hyp);
	saveMat("x.yml",x);
	*/

	/*
	cv::Mat x, y, cov, lik;
	hyp_infExact hyp;
	cv::FileStorage fs1("C:\\Users\\TZ\\Desktop\\C++GP\\hyp.cov1.yml", cv::FileStorage::READ);
	fs1["mat0000"] >> cov;
	hyp.cov.ell = cov.at<double>(0);
	hyp.cov.sf = cov.at<double>(1);
	fs1.release();
	cv::FileStorage fs2("C:\\Users\\TZ\\Desktop\\C++GP\\hyp.lik1.yml", cv::FileStorage::READ);
	fs2["mat0000"] >> lik;
	hyp.lik = lik.at<double>(0);
	fs2.release();
	cv::FileStorage fs3("C:\\Users\\TZ\\Desktop\\C++GP\\hyp.mean1.yml", cv::FileStorage::READ);
	fs3["mat0000"] >> hyp.mean;
	fs3.release();
	cv::FileStorage fs4("C:\\Users\\TZ\\Desktop\\C++GP\\x1.yml", cv::FileStorage::READ);
	fs4["mat0000"] >> x;
	fs4.release();
	cv::FileStorage fs5("C:\\Users\\TZ\\Desktop\\C++GP\\y1.yml", cv::FileStorage::READ);
	fs5["mat0000"] >> y;
	fs5.release();
	GPML gpml;
	Ans_infExact Ans;

	Ans = gpml.infExact(hyp, x, y, 3, Ans);
	*/
	


	/*
	cv::Mat x,cov;
	HPY_conSEiso shit;
	cv::FileStorage fs1("C:\\Users\\TZ\\Desktop\\C++GP\\hyp.cov1.yml", cv::FileStorage::READ);
	fs1["mat0000"] >> cov;
	shit.ell = cov.at<double>(0);
	shit.sf = cov.at<double>(1);
	fs1.release();
	cv::FileStorage fs4("C:\\Users\\TZ\\Desktop\\C++GP\\x1.yml", cv::FileStorage::READ);
	fs4["mat0000"] >> x;
	fs4.release();


	GPML gpml;
	Mat c;// = gpml.sq_dist(a, Mat());
	c = gpml.covSEiso(shit, x, "", 1 , 4);

	saveMat("C.yml", c);
	*/

    cv::Mat cov;
	hyp_infExact X;
	cv::FileStorage fs1("C:\\Users\\TZ\\Desktop\\C++GP\\X.cov.yml", cv::FileStorage::READ);
	fs1["mat0000"] >> cov;
	X.cov.ell = cov.at<double>(0);
	X.cov.sf = cov.at<double>(1);
	fs1.release();
	X.lik = 2.3025850929940455;
	cv::FileStorage fs2("C:\\Users\\TZ\\Desktop\\C++GP\\X.cov.yml", cv::FileStorage::READ);
	fs2["mat0000"] >> X.mean;
	fs2.release();

	cv::Mat mat_length = cv::Mat::Mat(1,1,cov.type());
	mat_length.at<double>(0) = -100;

	Ans_minimize Ans;
	GPML gpml;

	Ans = gpml.minimize(X,mat_length,Ans);
	
	

	return 0;
}