#pragma once
#include <opencv2\opencv.hpp>

using namespace std;

class HPY_conSEiso
{
public:
	HPY_conSEiso()
	{

	}
public:
	double ell;
	double sf;
};

class Ans_likGauss
{
public:
	cv::Mat ymu;
	cv::Mat ys2;
	cv::Mat lp;
	cv::Mat dlp;
	cv::Mat d2lp;
	cv::Mat d3lp;
	cv::Mat lZ;
	cv::Mat dlZ;
	cv::Mat d2lZ;
	cv::Mat dlZhyp;
	cv::Mat b;
	cv::Mat z;
	cv::Mat lp_dhyp;
	cv::Mat dlp_dhyp;
	cv::Mat d2lp_dhyp;
	Ans_likGauss()
	{
	}
};

class hyp_infExact
{
public:
    HPY_conSEiso cov;
	//cv::Mat lik;
	double lik;
	cv::Mat mean;
	hyp_infExact()
	{
	}
};

class post_infExact
{
public:
	cv::Mat alpha;
	cv::Mat sW;
	cv::Mat L;
	post_infExact()
	{
	}
};

class Ans_infExact
{
public:
	post_infExact post;
	cv::Mat nlZ;
	hyp_infExact dnlZ;
	Ans_infExact()
	{
	}
};

class Ans_minimize
{
public:
	hyp_infExact output_X;
	cv::Mat output_fX;
	double out_i;
	Ans_minimize()
	{
	}
};

class GPML
{
public:
	GPML(void);
	~GPML(void);


public:
	cv::Mat matlib_mean(cv::Mat &a, int oper);
	cv::Mat matlib_sum(cv::Mat &a, int oper);
	cv::Mat my_chol(cv::Mat &a);
	cv::Mat solve_chol(cv::Mat &L, cv::Mat &b);
	cv::Mat unwrap(hyp_infExact &s);
	hyp_infExact rewrap(cv::Mat &s);

public:
	cv::Mat sq_dist(cv::Mat &a, cv::Mat &b);
	cv::Mat covSEiso(HPY_conSEiso hpy, cv::Mat &x, std::string z = "", int i = 1, int nargin = 3);
	int covSEiso();
	
	cv::Mat likGauss(); //less than three inputs
	Ans_likGauss likGauss(cv::Mat &hyp, cv::Mat &y, cv::Mat &mu, int nargout, Ans_likGauss &Ans); //three inputs
	Ans_likGauss likGauss(cv::Mat &hyp, cv::Mat &y, cv::Mat &mu, cv::Mat &s2, int nargout, Ans_likGauss &Ans); //four inputs
	Ans_likGauss likGauss(cv::Mat &hyp, cv::Mat &y, cv::Mat &mu, cv::Mat &s2, std::string inf, int nargout, Ans_likGauss &Ans); //five inputs
	Ans_likGauss likGauss(cv::Mat &hyp, cv::Mat &y, cv::Mat &mu, cv::Mat &s2, std::string inf, char i, int nargout, Ans_likGauss &Ans); //six inputs

	cv::Mat meanConst(cv::Mat &hyp, cv::Mat &x, int i);
	cv::Mat meanConst(cv::Mat &hyp, cv::Mat &x);

	Ans_infExact infExact(hyp_infExact &hyp, cv::Mat &x, cv::Mat &y, int nargout, Ans_infExact &Ans);

	Ans_minimize minimize(hyp_infExact &X, cv::Mat &mat_length,/* cv::Mat &x, cv::Mat &y,*/ Ans_minimize &Ans);
};