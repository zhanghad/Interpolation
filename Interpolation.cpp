#include <opencv.hpp>
#include <string>
#include <vector>
#include "rbf.h"

using namespace cv;


Mat getScribbleMask(const Mat& image, const Mat& scribbles, double eps = 0, int nErosions = 0);
Mat deletePixel(Mat& imageSrc, Mat& imageDst, double ratio);
double metric_MSE(const Mat& original, const cv::Mat& processed);
double metric_PSNR(const Mat& original, const cv::Mat& processed);
double metric_SSIM(const Mat& imgx, const Mat& imgy);

Mat interpolation_nearest(const Mat& image, const Mat& mask,int flag);
Mat interpolation_rbf(const Mat& image, const Mat& mask, int flag, int neighbor_distance=1);


void test();
void test2();

int main()
{
    //test2();	
    //test();
    
    string image_dir = "image/";
    //for(int i=1;i<=4;i++) {
    //    cv::Mat image=cv::imread(imageDir + "test\\" + std::to_string(i) + ".jpg");
    //    cv::imwrite(imageDir + "test\\" + std::to_string(i) + ".bmp",image);
    //}


    int index = 1;
    Mat img_src = imread(image_dir + "test/" + to_string(index) + ".bmp");
    //Mat img_damage = imread(imageDir + "damage/" + to_string(index) + ".bmp");
    //Mat img_mask = getScribbleMask(img_src, img_damage);
	
    Mat img_lost;
	Mat img_lost_mask=deletePixel(img_src, img_lost, 0.90);//随机丢点

    //Mat img_result = interpolation_nearest(img_damage, img_mask, 1);
    Mat img_result=interpolation_rbf(img_lost, img_lost_mask, 1,10);
	
    //Mat img_result = interpolation_nearest(img_lost, img_lost_mask, 1);
    
    imshow("img_result", img_result);
    imshow("img_src", img_src);
    //imshow("img_damage", img_damage);
    //imshow("img_mask", img_mask);
    imshow("img_lost", img_lost);
    imshow("img_lost_mask", img_lost_mask);
    waitKey();

	
    //cout << metric_MSE(img_src, img_damage)<<endl;
    //cout << metric_PSNR(img_src, img_damage) << endl;
    //cout << metric_SSIM(img_src, img_damage) << endl;


    //imwrite(imageDir + "result\\" + to_string(index) + "_near_2.bmp", img_result);

    waitKey();
}


Mat getScribbleMask(const Mat& image, const Mat& scribbles, double eps, int nErosions)
{
    Mat diff;
    absdiff(image, scribbles, diff);
    vector<Mat> channels;
    split(diff, channels);
    Mat mask = channels[0] + channels[1] + channels[2];
    threshold(mask, mask, eps, 255, cv::THRESH_BINARY);//二值化，255为差异点
    erode(mask, mask, cv::Mat(), cv::Point(-1, -1), nErosions);//侵蚀
    return mask;
}

double metric_MSE(const Mat& original, const Mat& processed) {
    if (original.size() != processed.size()) {
        cout << "wrong size" << endl;
        return 0;
    }
    Mat temp(original.size(), CV_64FC3);
    subtract(original, processed, temp,noArray(), CV_64FC3);
    multiply(temp, temp, temp,1, CV_64FC3);
	Scalar temps = mean(temp);
    return (temps.val[0]+ temps.val[1] + temps.val[2])/3.0;
}

double metric_PSNR(const Mat& original, const Mat& processed) {
    if(original.size()!=processed.size()) {
        cout << "wrong size" << endl;
        return 0;
    }
    return (10 * log10(255 * 255 / metric_MSE(original, processed)));
}

double metric_SSIM(const Mat& imgx, const Mat& imgy) {
    //double k1 = 0.01;
    //double k2 = 0.03;
    //double c1 = 255 * 255 * k1 * k1;
    //double c2 = 255 * 255 * k2 * k2;
    const double C1 = 6.5025, C2 = 58.5225;

    Mat i1 = imgx;
    Mat i2 = imgy;

    int d = CV_32FC1;
    Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);

    Mat I2_2 = I2.mul(I2);
    Mat I1_2 = I1.mul(I1);
    Mat I1_I2 = I1.mul(I2);
    Mat mu1, mu2;
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
	//warn
    multiply(mu1_mu2, 2, t1);
    multiply(sigma12, 2, t2);
    t1 += C1;
    t2 += C2;
	
    t3 = t1.mul(t2);
    Mat temp = mu1_2 + mu2_2;
    t1 = temp + C1;
    Mat temp2 = sigma1_2 + sigma2_2;
    t2 = temp2 + C2;
    t1 = t1.mul(t2);
    Mat ssim_map;
    divide(t3, t1, ssim_map);
    Scalar mssim = mean(ssim_map);
    double result = mssim.val[0];
    return result;
}



//最近邻插值
Mat interpolation_nearest(const Mat&image, const Mat& mask,int flag) {
    if(image.size!=mask.size) {
        cout << "size does not match" << endl;
    }
	
    Mat result = image.clone();

	for(int i=0;i<result.rows;i++) {
        for (int j = 0; j < result.cols; j++) {
            if(mask.at<uchar>(i,j)==255) {
                //城市距离
                if(flag==1) {
                    bool found = false;
                	//由近至远寻找
                    for (int distance = 1;; distance++) {
                        //上
                        if (i - distance >= 0) {
                            for (int y1 = i - distance; y1 <= i + distance; y1++) {
                                if (y1 < 0)
                                    continue;
                                if (y1 >= result.cols)
                                    break;
                                if (mask.at<uchar>(i - distance, y1) == 0) {
                                    result.at<Vec3b>(i, j) = image.at<Vec3b>(i - distance, y1);
                                    found = true;
                                    break;
                                }

                            }
                        }
                        if (found)
                            break;

                        //下
                        if (i + distance < result.rows) {
                            for (int y2 = i - distance; y2 <= i + distance; y2++) {
                                if (y2 < 0)
                                    continue;
                                if (y2 >= result.cols)
                                    break;
                                if (mask.at<uchar>(i + distance, y2) == 0) {
                                    result.at<Vec3b>(i, j) = image.at<Vec3b>(i + distance, y2);
                                    found = true;
                                    break;
                                }

                            }
                        }
                        if (found)
                            break;

                        //右
                        if (j + distance < result.cols) {
                            for (int x2 = i - distance + 1; x2 <= i + distance - 1; x2++) {
                                if (x2 < 0)
                                    continue;
                                if (x2 >= result.rows)
                                    break;
                                if (mask.at<uchar>(x2, j + distance) == 0) {
                                    result.at<Vec3b>(i, j) = image.at<Vec3b>(x2, j + distance);
                                    found = true;
                                    break;
                                }

                            }

                        }
                        if (found)
                            break;

                        //左
                        if (j - distance >= 0) {
                            for (int x1 = i - distance + 1; x1 <= i + distance - 1; x1++) {
                                if (x1 < 0)
                                    continue;
                                if (x1 >= result.rows)
                                    break;
                                if (mask.at<uchar>(x1, j - distance) == 0) {
                                    result.at<Vec3b>(i, j) = image.at<Vec3b>(x1, j - distance);
                                    found = true;
                                    break;
                                }


                            }

                        }
                        if (found)
                            break;
                	
                }

                }

            }
        }
	}

    return result;
}


Mat interpolation(const Mat& image, const Mat& mask, int flag) {
    if (image.size != mask.size) {
        cout << "size does not match" << endl;
    }

    Mat result = image.clone();

    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            if (mask.at<uchar>(i, j) == 255) {



            }
        }
    }
    return result;
}


Mat interpolation_bilinear(const Mat& image, const Mat& mask, int flag) {
    cout << "rbf interpolating ..." << endl;
	
    if (image.size != mask.size) {
        cout << "size does not match" << endl;
    }

    Mat result = image.clone();

    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            if (mask.at<uchar>(i, j) == 255) {



            }
        }
    }
    return result;
}


Mat interpolation_rbf(const Mat& image, const Mat& mask, int flag,int neighbor_distance) {
    cout << "rbf interpolating ..." << endl;
	
    if (image.size != mask.size) {
        cout << "size does not match" << endl;
    }
    Mat result = image.clone();


    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
        	//判断是否需要插值
            if (mask.at<uchar>(i, j) == 255) {
                //遍历邻域

			    int exist_count = 0;
			    //int pre_count = 0;            	

            	//统计数量
                for (int x = i - neighbor_distance; x < i + neighbor_distance; x++) {
                    if (x < 0 || x >= result.rows)
                        continue;

                    for (int y = j - neighbor_distance; y < j + neighbor_distance; y++) {
                        if (y < 0 || y >= result.cols)
                            continue;
                        if (mask.at<uchar>(x, y) != 255)
                            exist_count++;
                        //else
                        //    pre_count++;

                    }
                }

            	//可参考的数量过少
            	if(exist_count==0) {
                    cout << "neighbor distance is too small" << endl;
            		break;
            	}
                    

            	//分配矩阵
                MatrixXd exist_pos(exist_count,2);
                MatrixXd exist_b_value(exist_count, 1);
                MatrixXd exist_g_value(exist_count, 1);
			    MatrixXd exist_r_value(exist_count,1);

                vector<MatrixXd> exist_value;
                exist_value.push_back(exist_b_value);
                exist_value.push_back(exist_g_value);
                exist_value.push_back(exist_r_value);
            	
			    //MatrixXd pre_pos(pre_count,2);
                MatrixXd pre_pos(1, 2);
			    MatrixXd pre_value;

                pre_pos << i, j;
            	

                int index_exist = 0;
				//int index_pre = 0;

            	//填充矩阵
                for (int x = i - neighbor_distance; x < i + neighbor_distance; x++) {
                    if (x < 0 || x >= result.rows)
                        continue;

                    for (int y = j - neighbor_distance; y < j + neighbor_distance; y++) {
                        if (y < 0 || y >= result.cols)
                            continue;
                        if (mask.at<uchar>(x, y) != 255) {
                            exist_pos(index_exist, 0) = x;
                            exist_pos(index_exist, 1) = y;
                        	
                            exist_value[0](index_exist, 0) = result.at<Vec3b>(x, y)[0];
                            exist_value[1](index_exist, 0) = result.at<Vec3b>(x, y)[1];
                            exist_value[2](index_exist, 0) = result.at<Vec3b>(x, y)[2];
                            index_exist++;
                        }   
                        //else {
                        //    pre_pos(index_pre, 0) = x;
                        //    pre_pos(index_pre, 1) = y;
                        //    index_pre++;
                        //}

                    }
                }

            	//计算未知值
                //cout << "solve" << endl;
                for(int index=0;index<3;index++) {
                    MatrixXd rbfcoeff = rbfcreate(exist_pos, exist_value[index], rbfphi_multiquadrics, 0.4444, 0);
                    pre_value = rbfinterp(exist_pos, rbfcoeff, pre_pos, rbfphi_multiquadrics, 0.4444);
                    //cout << pre_value(0, 0) << endl;
                    result.at<Vec3b>(i, j)[index] = (int)(pre_value(0, 0));
                }

            }
        }
    }

    return result;
}


Mat deletePixel(Mat& imageSrc, Mat& imageDst, double ratio) {
	
    imageDst = imageSrc.clone();
    int thres = (int)(ratio * 255);
    Mat mask = Mat(imageDst.size(), CV_8UC1);
    randu(mask, Scalar::all(0), Scalar::all(255));
    threshold(mask, mask, thres, 255, cv::THRESH_BINARY);
    bitwise_not(mask, mask);//255为丢掉的点
    for(int i=0;i<imageDst.rows;i++) {
        for(int j=0;j<imageDst.cols;j++) {
            if(mask.at<uchar>(i,j)==255) {
                imageDst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            }
        }
    }
    return mask;
}


void test() {
    string imageColoredDir = "D:\\Visual_studio_file\\Colorization_console\\image\\colorize\\";
    string imageSrcDir = "D:\\Visual_studio_file\\Colorization_console\\image\\original\\";


    int imageIndex = 4;
    int expIndex = 1;

	for(;expIndex<=3;expIndex++) {
        string imageColoredName = "test" + to_string(imageIndex) + "_result_n" + to_string(expIndex) + ".bmp";
        string imageSrcName = "test" + to_string(imageIndex) + "_original.bmp";

        Mat imageSrc = imread(imageSrcDir + imageSrcName);
        Mat imageColored = imread(imageColoredDir + imageColoredName);

        cout << "MSE:" << metric_MSE(imageSrc, imageColored) << endl;
        cout << "PSNR:" << metric_PSNR(imageSrc, imageColored) << endl;
        cout << "SSIM:" << metric_SSIM(imageSrc, imageColored) << endl;

	}
	
}

void test2(){
    //MatrixXd m1(9, 1);
    //m1 << -2.0,
    //    -1.5,
    //    -1.0,
    //    -0.5,
    //    0,
    //    0.5,
    //    1.0,
    //    1.5,
    //    2.0;
	MatrixXd m1(5, 2);
    m1 <<0,0,
		0,1,
		1,0,
		1,2,
		2,2;
    MatrixXd m2(5, 1);
    m2 <<255,
	150,
	150,
	70,
	50;
    //MatrixXd m3(401, 1);
    //for (int i = 0; i < 401; i++)
    //{
    //    m3(i) = -2 + (double)i * 0.01;
    //}
	//MatrixXd m3(401, 2);
 //   for (int i = 0; i < 401; i++)
 //   {
 //       m3(i,0) = -2 + (double)i * 0.01;
 //       m3(i, 1) = 2 + (double)i * 0.01;
 //   }

    MatrixXd m3(4, 2);
    m3 << 0, 2,
        1, 1,
        2, 0,
        2, 1;
	
    //cout << "m1" << endl<< m1 << endl;
    //cout << "m2" << endl << m2 << endl;
    MatrixXd rbfcoeff;
    rbfcoeff = rbfcreate(m1, m2, rbfphi_multiquadrics, 0.4444, 0);
    MatrixXd result;
    result = rbfinterp(m1, rbfcoeff, m3, rbfphi_multiquadrics, 0.4444);
    //cout << "result" << endl << result << endl;

    //cout << "check" << endl << rbfcheck(m1, m2, rbfcoeff, rbfphi_multiquadrics, 0.4444) << endl;

    cout << result << endl;
    cout << result.cols() <<" "<<result.rows()<<endl;
	
	
    cout << (m1 - m2).cwiseAbs().maxCoeff() << endl;

}