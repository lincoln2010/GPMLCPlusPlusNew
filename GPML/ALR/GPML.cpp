#include "StdAfx.h"
#include "GPML.h"
#include <iostream>
#include<iomanip>
#include<cmath>
using namespace std;
#define isnan(x) _isnan(x)
#define isinf(x) (!_finite(x))
#define M_PI       3.14159265358979323846
#define my_max(x,y)  ( x>y?x:y )
#define my_min(x,y)  ( x>y?y:x )

GPML::GPML(void)
{
}


GPML::~GPML(void)
{
}


cv::Mat GPML::matlib_mean(cv::Mat &a, int oper)
{
	CV_Assert(a.type() == CV_64F);

	if (oper == 2)
	{
		//按行
		cv::Mat ret(a.rows,1, a.type());
		for (int i = 0; i < a.rows; i++)
		{
			ret.at<double>(i) = cv::mean(a.row(i))[0];
		}
		return ret;
	}
	else if (oper == 1)
	{
		//按列
		cv::Mat ret(1, a.cols, a.type());
		for (int i = 0; i < a.cols; i++)
		{
			ret.at<double>(i) = cv::mean(a.col(i))[0];
		}
		return ret;
	}
	else
	{
		CV_Assert(0);
		
	}
	return cv::Mat();
}


cv::Mat GPML::matlib_sum(cv::Mat &a , int oper)
{
	CV_Assert(a.type() == CV_64F);

	if (oper == 2)
	{
		//按行
		cv::Mat ret(a.rows, 1, a.type());
		for (int i = 0; i < a.rows; i++)
		{
			ret.at<double>(i) = cv::sum(a.row(i))[0];
		}
		return ret;
	}
	else if (oper == 1)
	{
		//按列
		cv::Mat ret(1, a.cols, a.type());
		for (int i = 0; i < a.cols; i++)
		{
			ret.at<double>(i) = cv::sum(a.col(i))[0];
		}
		return ret;
	}
	else
	{
		CV_Assert(0);
		
	}
	return cv::Mat();
}


cv::Mat GPML::my_chol(cv::Mat &A)
{
	if(A.rows != A.cols || A.rows == 0)
	{
		CV_Assert(0);
	}
	int n = A.rows;
	cv::Mat L = cv::Mat::zeros(n, n, A.type());
	for(int i = 0; i <n ; i++)
		for(int j = 0; j < i+1; j++)
		{
			double s = 0;
			for(int k = 0; k < j; k++)
			{
				s = s + L.at<double>(i,k) * L.at<double>(j,k);
			}
			if(i == j)
			{
				L.at<double>(i,j) = sqrt(A.at<double>(i,i) - s);
			}
			else
			{
				L.at<double>(i,j) = (1.0 / L.at<double>(j,j) * (A.at<double>(i,j) - s));
			}
		}
	return L;
}


cv::Mat GPML::solve_chol(cv::Mat &L, cv::Mat &b)
{
	double t1 = L.rows;
	double t2 = L.cols;
	double t3 = b.rows;
	if (t1 != t2 || t1 != t3)
	{
		CV_Assert(0);
	}
	cv::Mat temp1 = L.t();
	cv::Mat temp2 = temp1.inv();
	cv::Mat temp3 = L.inv();
	cv::Mat Ans;
	Ans = temp3 * temp2 * b;
	return Ans;
}


cv::Mat GPML::unwrap(hyp_infExact &s)
{
	cv::Mat Ans = cv::Mat::Mat(4,1,s.mean.type());
	Ans.at<double>(0) = s.cov.ell;
	Ans.at<double>(1) = s.cov.sf;
	Ans.at<double>(2) = s.lik;
	Ans.at<double>(3) = s.mean.at<double>(0);
	return Ans;
}


hyp_infExact GPML::rewrap(cv::Mat &s)
{
	hyp_infExact Ans;
	Ans.cov.ell = s.at<double>(0);
	Ans.cov.sf = s.at<double>(1);
	Ans.lik = s.at<double>(2);
	Ans.mean = cv::Mat::Mat(1,1,s.type());
	Ans.mean.at<double>(0) = s.at<double>(3);
	return Ans;
}


cv::Mat GPML::sq_dist(cv::Mat &a, cv::Mat &b)
{
	CV_Assert(!a.empty());

	int D = a.rows;
	int n = a.cols;

	int m;
	if (b.empty())
	{
		cv::Mat mu = this->matlib_mean(a, 2);
		mu = cv::repeat(mu, 1, a.cols);

		a = a - mu;

		b = a;
		m = n;
	}
	else
	{
		CV_Assert(0);
		int d = b.rows;
		m = b.cols;

		CV_Assert(d == D);

		cv::Mat mu = (m/(m+n)) * matlib_mean(b,2) + (n/(m+n)) * matlib_mean(a,2);

		a = a - cv::repeat(mu, 1, n);

		b = b - cv::repeat(mu, 1, m);
	}
	cv::Mat aa = a.mul(a);
	cv::Mat sum_aa = matlib_sum(aa, 1).t();
	

	cv::Mat bb = b.mul(b);
	cv::Mat sum_bb = matlib_sum(bb, 1);


	cv::Mat C = cv::repeat(sum_aa, 1, m) + cv::repeat(sum_bb, n, 1) - 2*a.t() * b;

	C = cv::max(C,0);

	return C;

}

int GPML::covSEiso()
{
	return '2';
}

cv::Mat GPML::covSEiso(HPY_conSEiso hpy, cv::Mat &x, std::string z, int i, int nargin)
{
	CV_Assert(!x.empty());

	cv::Mat K;
	bool xeqz = !z.length();

	int dg = z.length() ==0 ? 0: strcmp(z.c_str(), "diag");
	
	double ell = exp(hpy.ell);

	

	double sf2 = exp(2*hpy.sf);

	if (dg)
	{
		K = cv::Mat(x.rows, 1, x.type());
	}
	else
	{
		if (xeqz)
		{
			cv::Mat t = x.t() / ell;
			K = sq_dist(t, cv::Mat());
		}
		else
		{
			CV_Assert(0);
			//K = sq_dist(x'/ell,z'/ell);, unsupport
		}

		//if (z.length() == 0 || i == 0)
		if(nargin < 4)
		{
			cv::Mat T;
			cv::exp(-1.0/2 * K, T);
			K = sf2 * T;
		}
		else if(i == 1)
		{
			cv::Mat T;
			cv::exp(-1.0/2 * K, T);

			K = sf2 * T.mul(K);

		}
		else if (i == 2)
		{
			cv::Mat T;
			cv::exp(-1.0/2 * K, T);
			K = 2 * sf2 * T;
		}
		else
		{
			//unconver m to c
			CV_Assert(0);
		}

	}

	return K;
		
}

Ans_likGauss GPML::likGauss(cv::Mat &hyp, cv::Mat &y, cv::Mat &mu, int nargout,Ans_likGauss &Ans)
{
	cv::Mat sn2 = cv::Mat(hyp.rows, hyp.cols, hyp.type()); 
	cv::exp(2*hyp,sn2);

	if (y.empty() == 1)
	{
		y = cv::Mat::zeros(mu.rows, mu.cols, mu.type());
	}
	double s2zeros = 1;
	cv::Mat temp1;
	cv::log(2*M_PI*sn2,temp1);
	Ans.lp = - (y - mu)*(y - mu)/sn2/2 - temp1/2;
	if (nargout > 1)
	{
		Ans.ymu = mu;
	}
	if (nargout > 2)
	{
		Ans.ys2 = sn2;
	}
	return Ans;
}

Ans_likGauss GPML::likGauss(cv::Mat &hyp, cv::Mat &y, cv::Mat &mu, cv::Mat &s2, int nargout, Ans_likGauss &Ans)
{
	cv::Mat sn2 = cv::Mat(hyp.rows, hyp.cols, hyp.type()); 
	cv::exp(2*hyp,sn2);

	if (y.empty() == 1)
	{
		y = cv::Mat::zeros(mu.rows, mu.cols, mu.type());
	}
	double s2zero = 1;
	if (cv::norm(s2) > 0)
	{
		s2zero = 0;
	}
	if (s2zero == 1)
	{
		cv::Mat temp1;
		cv::log(2*M_PI*sn2,temp1);
		Ans.lp = - (y - mu)*(y - mu)/sn2/2 - temp1/2;
		s2 = 0;
	}
	else
	{
		Ans = likGauss(hyp,y,mu,s2,"infEP",nargout-2,Ans);
	}
	if (nargout > 1)
	{
		Ans.ymu = mu;
	}
	if( nargout > 2)
	{
		Ans.ys2 = sn2 + s2;
	}
	return Ans;
}

Ans_likGauss GPML::likGauss(cv::Mat &hyp, cv::Mat &y, cv::Mat &mu, cv::Mat &s2, std::string inf, int nargout, Ans_likGauss &Ans)
{
	cv::Mat sn2 = cv::Mat(hyp.rows, hyp.cols, hyp.type()); 
	cv::exp(2*hyp,sn2);

	if (inf == "infLaplace")
	{
		cv::Mat ymmu = y - mu;
		cv::Mat temp1;
		cv::log(2*M_PI*sn2,temp1);
		Ans.lp = -ymmu*ymmu/2/sn2 - temp1/2;
		if (nargout > 1)
		{
			Ans.dlp = ymmu/sn2;
		}
		if(nargout > 2)
		{
			cv::Mat temp2 = cv::Mat::ones(ymmu.rows, ymmu.cols, ymmu.type()); 
			Ans.d2lp = -temp2/sn2;
		}
		if(nargout > 3)
		{
			Ans.d3lp = cv::Mat::zeros(ymmu.rows, ymmu.cols, ymmu.type());
		}
		return Ans;
	}
	else if (inf == "infEP")
	{
		cv::Mat temp3;
		cv::log(2*M_PI*(s2 + sn2),temp3);
		Ans.lZ = -(y - mu)*(y - mu)/(s2 + sn2)/2 - temp3/2;
		if (nargout > 1)
		{
			Ans.dlZ = (y - mu)/(sn2 + s2);
		}
		if (nargout > 2)
		{
			Ans.d2lZ = -1/(sn2 + s2);
		}
		return Ans;
	}
	else if (inf == "infVB")
	{
		double num = s2.total();
		Ans.b = cv::Mat::zeros(num, 1, s2.type());
		cv::Mat temp4 = cv::Mat::ones(num, 1, s2.type());
		Ans.z = y * temp4;
		return Ans;
	}
	else
	{
		return Ans;
	}
}

Ans_likGauss GPML::likGauss(cv::Mat &hyp, cv::Mat &y, cv::Mat &mu, cv::Mat &s2, std::string inf, char i, int nargout, Ans_likGauss &Ans)
{
	cv::Mat sn2 = cv::Mat(hyp.rows, hyp.cols, hyp.type()); 
	cv::exp(2*hyp,sn2);

	if (inf == "infLaplace")
	{
		Ans.lp_dhyp = (y - mu)*(y - mu)/sn2 - 1;
		Ans.dlp_dhyp = 2*(mu - y)/sn2;
		cv::Mat temp1 = cv::Mat::ones(mu.rows, mu.cols, mu.type());
		Ans.d2lp_dhyp = 2*temp1/sn2;
		return Ans;
	}
	else if (inf == "infEP")
	{
		Ans.dlZhyp = ((y - mu)*(y - mu)/(sn2 + s2) - 1)/(1+s2/sn2);
		return Ans;
	}
	else if (inf == "infVB")
	{
		double num = s2.total();
		Ans.b = cv::Mat::zeros(num, 1, s2.type());
		cv::Mat temp4 = cv::Mat::ones(num, 1, s2.type());
		Ans.z = y * temp4;
		return Ans;
	}
	else
	{
		return Ans;
	}
}

cv::Mat GPML::meanConst(cv::Mat &hyp, cv::Mat &x)
{
	if(hyp.total() != 1)
	{
		CV_Assert(0);
	}
	cv::Mat C = hyp;
	cv::Mat temp = cv::Mat::ones(x.rows,1,x.type());
	double tempt = C.at<double>(0,0);
	cv::Mat A = tempt * temp;
	return A;
}

cv::Mat GPML::meanConst(cv::Mat &hyp, cv::Mat &x,int i)
{
	if(hyp.total() != 1)
	{
		CV_Assert(0);
	}
	cv::Mat C = hyp;
	cv::Mat A;
	if(i == 1)
	{
		A = cv::Mat::ones(x.rows,1,x.type());
	}
	else
	{
		A = cv::Mat::zeros(x.rows,1,x.type());
	}
	return A;
}

Ans_infExact GPML::infExact(hyp_infExact &hyp, cv::Mat &x, cv::Mat &y, int nargout, Ans_infExact &Ans)
{
	double n = x.rows;
	double D = x.cols;
	cv::Mat K, m, tempL, pL;
	K = covSEiso(hyp.cov, x);
	//cout<<"K = "<<K.at<double>(0,0)<<" "<<K.at<double>(0,1)<<" "<<K.at<double>(1,0)<<" "<<K.at<double>(1,1)<<endl;
	m = meanConst(hyp.mean, x);
	//cout<<"m = "<<m<<endl;

	double sn2 = exp(2*hyp.lik);

	cv::Mat temp1 =	cv::Mat::eye(n, n, x.type());
	cv::Mat temp2 = cv::Mat::ones(n, 1, x.type());
	double sl = 0;
	if (sn2 < 1E-6)
	{
		cv::Mat Temp1 = K + sn2*temp1;
		tempL = my_chol(Temp1);
		sl = 1;
		pL = -solve_chol(tempL, temp1);
	}
	else
	{
		cv::Mat Temp2 = K/sn2 + temp1;
		tempL = my_chol(Temp2);
		sl = sn2;
		pL = tempL.t();
		//cout<<"pL = "<<pL.at<double>(0)<<" "<<pL.at<double>(1)<<" "<<pL.at<double>(2)<<" "<<pL.at<double>(3)<<endl;
	}
	cv::Mat Temp3 = y - m;
	Ans.post.alpha = solve_chol(pL, Temp3)/sl;
	//cout<<"alpha = "<<Ans.post.alpha.at<double>(0)<<" "<<Ans.post.alpha.at<double>(1)<<endl;
	Ans.post.sW = temp2/sqrt(sn2);
	//cout<<"sW = "<<Ans.post.sW.at<double>(0)<<" "<<Ans.post.sW.at<double>(1)<<endl;
	Ans.post.L = pL;

	if (nargout > 1)
	{
		cv::Mat tt1, tt2, tt3;
		tt1 = (y - m).t()*Ans.post.alpha/2;
		tt2 = tempL.diag(0);
		cv::log(tt2,tt2);
		tt2 = matlib_sum(tt2, 1);
		cv::log(2*M_PI*sl, tt3);
		Ans.nlZ = tt1 + tt2 + n*tt3/2;
		//cout<<tt1<<" "<<tt2<<" "<<tt3<<" "<<"nlZ = "<<Ans.nlZ.at<double>(0)<<endl;
	}
	if (nargout > 2)
	{
		Ans.dnlZ = hyp;
		cv::Mat Q = solve_chol(pL,temp1)/sl - Ans.post.alpha * Ans.post.alpha.t();
		//cout<<"Q = "<<Q.at<double>(0,0)<<" "<<Q.at<double>(0,1)<<" "<<Q.at<double>(1,0)<<" "<<Q.at<double>(1,1)<<endl;
		for (int i = 1; i < 3; i ++)
		{

			cv::Mat tempK = covSEiso(hyp.cov, x, "", i, 4);
			//cout<<tempK<<endl;
			cv::multiply(tempK, Q, tempK);
			//cout<<tempK<<endl;
			if(i == 1)
			{
				Ans.dnlZ.cov.ell = matlib_sum(matlib_sum(tempK,1), 2).at<double>(0)/2;
			}
			else
			{
				Ans.dnlZ.cov.sf = matlib_sum(matlib_sum(tempK,1), 2).at<double>(0)/2;
			}
		}
		//cout<<Ans.dnlZ.cov.ell<<" "<<Ans.dnlZ.cov.sf<<endl;
		int Q_rows = Q.rows; 
		double Q_trace = 0;
		for(int i = 0; i < Q_rows; i++)
		{
			Q_trace = Q_trace + Q.at<double>(i,i);
		}
		Ans.dnlZ.lik = sn2 * Q_trace;
		double mean_numel = hyp.mean.total();
		for (int i = 1; i < mean_numel+1; i++)
		{
			cv::Mat tempm = meanConst(hyp.mean, x, i);
			tempm = tempm.t();
			tempm = tempm * Ans.post.alpha;
			//cout<<tempm;
			
			Ans.dnlZ.mean.at<double>(i-1) = tempm.at<double>(0);
		}
	}
	return Ans;
}

Ans_minimize GPML::minimize(hyp_infExact &X, cv::Mat &mat_length,/* cv::Mat &x, cv::Mat &y,*/ Ans_minimize &Ans)
{
	double INT =0.1, EXT = 3.0, MAX = 20, RATIO = 10, SIG = 0.1, RHO = 0.05;
	double length = 0;
	double red;
	if(mat_length.rows == 2 || mat_length.cols == 2)
	{
		length = mat_length.rows;
		red = mat_length.cols;
	}
	else
	{
		red = 1;
		length = mat_length.at<double>(0);
	}
	string S;
	if(length > 0)
		S = "Linesearch";
	else
		S = "Function evaluation";

	int i = 0, ls_failed = 0;
	//[f0 df0] = feval(f, X, varargin{:});    //ALRadvanced函数,输入是四个函数句柄加两个矩阵(形参里的x、y矩阵)
	double f0 = 8.9850636280274916;
	hyp_infExact df0, Z = X;
	
	
	//df0.lik = 0.0011150760463067236;
	//df0.cov.ell = -0.99557118735150918;
	//df0.cov.sf =  1.9956667230471794;
	//df0.mean = cv::Mat::Mat(1,1,X.mean.type());
	//df0.mean.at<double>(0)= -0.0000061585580881148155;
	
	
	
	cv::Mat unwrap_X = unwrap(X);        //X由结构体变为了4×1矩阵,以后对X的操作均变为对unwrap_X的操作
	cv::Mat unwrap_df0 = unwrap(df0);    //df0由结构体变为了4×1矩阵，以后对X的操作均变为对unwrap_df0的操作
	//cout<<unwrap_X<<endl;
	//cout<<unwrap_df0<<endl;
	
	cout<<S<<"\t\t"<<i<<";\tValue ";
	cout<<setprecision(7)<<setiosflags(ios::scientific)<<f0<<endl;
	fflush(stdout);
	//带unwrap前缀的均为矩阵，不带unwrap前缀的为hyp_infExact结构体
	double fX = f0;
	i = i + (length < 0);
	cv::Mat s = -unwrap_df0;
	cv::Mat temp_d0 = - s.t() * s;
	double d0 = temp_d0.at<double>(0);
	double x3 = red / (1 - d0);

	//各种变量
	cv::Mat unwrap_X0 = unwrap_X;
	cv::Mat unwrap_dF0 = unwrap_df0;
	double F0 = 0;
	double M = 0;
	double x2 = 0;
	double f2 = 0;
	double d2 = 0;
	double f3 = 0;
	cv::Mat unwrap_df3 = unwrap_df0;
	double success = 0;
	hyp_infExact df3;
	double d3 = 0;
	double x1 = 0;
	double f1 = 0;
	double d1 = 0;
	double x4 = 0;
	double f4 = 0;
	double d4 = 0;
	cv::Mat output_fX = cv::Mat::zeros(20,1,X.mean.type());
	output_fX.at<double>(0) = fX;
	cout<<output_fX<<endl;
	int fX_flag = 1;

	while (i < abs(length) )
	{
		i = i + (length > 0);

		unwrap_X0 = unwrap_X;
		unwrap_dF0 = unwrap_df0;
		F0 = f0;
		if(length > 0)
		{
			M = MAX;
		}
		else
		{
			M = ((MAX + length + i) < 0)?(MAX):(-length - i);
		}

		while(1)
		{
			x2 = 0;
			f2 = f0;
			d2 = d0;
			f3 = f0;
			unwrap_df3 = unwrap_df0;
			success = 0;
			while((success != 0) && (M > 0))
			{
				try
				{
					M = M -1;
					i = i + (length < 0);
					//[f3 df3] = feval(f, rewrap(Z,X+x3*s), varargin{:}); //此操作后，df3由矩阵变为结构体
					//hyp_infExact df3;
					unwrap_df3 = unwrap(df3);  //df3由结构体变为了4×1矩阵
					if(isnan(f3) || isinf(f3))
						throw 1;
					for(int k = 0; k < 4; k++)
						if(isnan(unwrap_df3.at<double>(k)) || isinf(unwrap_df3.at<double>(k)))
							throw 1;
					success = 1;
				}
				catch(int e)
				{
					x3 = (x2 + x3) / 2;
				}
			}

			if(f3 < F0)
			{
				unwrap_X0 = unwrap_X + x3*s;
				F0 = f3;
				unwrap_dF0 = unwrap_df3;
			}
			cv::Mat temp_d3 = unwrap_df3.t() * s;
			d3 = temp_d3.at<double>(0);

			if(d3 > SIG*d0 || f3 > f0+x3*RHO*d0 || M == 0)
				break;

			x1 = x2, f1 = f2, d1 = d2;
			x2 = x3; f2 = f3; d2 = d3;

			double A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);
			double B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
			x3 = x1-d1*pow((x2-x1),2)/(B+sqrt(B*B-A*d1*(x2-x1)));
			
			if(x3 < 0)
				x3 = x2 * EXT;
			else if(x3 > x2 * EXT)
				x3 = x2 * EXT;
			else if(x3 < x2 + INT*(x2-x1))
				x3 = x2 + INT*(x2-x1);
		}

		while(  (abs(d3) > -SIG * d0 || f3 > f0+x3*RHO*d0) && M > 0)
		{
			if(d3 > 0 || f3 > f0+x3*RHO*d0)
			{
				x4 = x3;
				f4 = f3;
				d4 = d3;
			}
			else
			{
				x2 = x3;
				f2 = f3;
				d2 = d3;
			}

			if(f4 > f0)
				x3 = x2-(0.5*d2*pow((x4-x2),2))/(f4-f2-d2*(x4-x2));
			else
			{
				double A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);
				double B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
				x3 = x2+(sqrt(B*B-A*d2*pow((x4-x2),2))-B)/A;
			}
			if(isnan(x3) || isinf(x3))
				x3 = (x2+x4)/2;

			x3 = my_max(my_min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));
			//[f3 df3] = feval(f, rewrap(Z,X+x3*s), varargin{:}); //df3由矩阵变回结构体
			unwrap_df3 = unwrap(df3);  //df3由结构体变为了4×1矩阵
			if(f3 < F0)
			{
				unwrap_X0 = unwrap_X + x3*s;
				F0 = f3;
				unwrap_dF0 = unwrap_df3;
			}
			M = M - 1;
			i = i + (length < 0);
			cv::Mat temp_d3 = unwrap_df3.t() * s;
			d3 = temp_d3.at<double>(0);
		}

		if(abs(d3) < SIG*d0 && f3 < f0+x3*RHO*d0)
		{
			unwrap_X = unwrap_X + x3 * s;
			f0 = f3;
			output_fX.at<double>(fX_flag) = f0;
			fX_flag++;
			cout<<S<<"\t\t"<<i<<";\tValue ";
			cout<<setprecision(7)<<setiosflags(ios::scientific)<<f0<<endl;
			fflush(stdout);

			s = (unwrap_df3.t()*unwrap_df3 - unwrap_df0.t()*unwrap_df3)/(unwrap_df0.t()*unwrap_df0) * s - unwrap_df3;
			unwrap_df0 = unwrap_df3;
			d3 = d0;
			temp_d0 = unwrap_df0.t() * s;
			d0 = temp_d0.at<double>(0);
			if(d0 > 0)
			{
				s = -unwrap_df0;
				temp_d0 = - s.t() * s;
				d0 = temp_d0.at<double>(0);
			}
			x3 = x3 * my_min(RATIO, d3/d0);
			ls_failed = 0;
		}
		else
		{
			unwrap_X = unwrap_X0;
			f0 = F0;
			unwrap_df0 = unwrap_dF0;
			if( ls_failed || i > abs(length) )
				break;

			s = - unwrap_df0;
			temp_d0 = - s.t() * s;
			d0 = temp_d0.at<double>(0);
			x3 = 1/(1 - d0);
			ls_failed = 1;
		}
	}
	//X = rewrap(Z,unwrap_X); //X由矩阵变回结构体
	Ans.output_fX = output_fX;
	Ans.output_X = X;
	Ans.out_i = i;
	return Ans;
}