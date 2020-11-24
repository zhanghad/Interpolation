//张浩东 3018201404
#include <opencv.hpp>
#include <string>
#include <vector>
#include "rbf.h"

using namespace cv;


//评价方法
double metric_MSE(const Mat& original, const Mat& processed);
double metric_PSNR(const Mat& original, const Mat& processed);
double metric_SSIM(const Mat& imgx, const Mat& imgy);

//插值方法
Mat interpolation_nearest(const Mat& image, const Mat& mask, int flag);//flag=0为8邻域 flag=1为4邻域
Mat interpolation_rbf(const Mat& image, const Mat& mask, MatrixXd RBFFunction(MatrixXd r, double const_num), int neighbor_distance = 5);
Mat interpolation_rbf_new(const Mat& image, const Mat& mask, MatrixXd RBFFunction(MatrixXd r, double const_num), int neighbor_distance = 5);
Mat interpolation_bilinear(const Mat& image, const Mat& mask, int flag);//flag=0为8邻域 flag=1为4邻域

//获取mask
Mat getScribbleMask(const Mat& image, const Mat& scribbles, double eps = 0, int nErosions = 0);
//随机删除像素
Mat deletePixel(const Mat& imageSrc, Mat& imageDst, double ratio);


int main()
{
    //图像目录路径
    string image_src_dir = "image/original/";
    string image_damage_dir = "image/damage/";
    string image_reult_dir = "image/result/";
    string image_mask_dir = "image/mask/";

    
    //task 1 Random scribbles
    cout << "task 1 Random scribbles" << endl;
	for(int index = 1;index<=4;index++) {
        cout << "image " << to_string(index) << " scribbles" << endl;
		//读取原图
        Mat img_src = imread(image_src_dir + to_string(index) + ".bmp");
        Mat img_damage = imread(image_damage_dir + "scribbles/" + to_string(index) + "_scribbles_1.bmp");
        Mat img_mask = getScribbleMask(img_src, img_damage);
		//修复
        Mat img_result_near = interpolation_nearest(img_damage, img_mask, 0);//8邻域
        //Mat img_result_near_1 = interpolation_nearest(img_damage, img_mask, 1);//4邻域
        Mat img_result_bilinear = interpolation_bilinear(img_damage, img_mask,0);//8邻域
        Mat img_result_rbf_mq = interpolation_rbf(img_damage, img_mask, rbfphi_multiquadrics, 8);
        Mat img_result_rbf_tps = interpolation_rbf(img_damage, img_mask, rbfphi_thinplate, 8);
        Mat img_result_rbf_g = interpolation_rbf(img_damage, img_mask, rbfphi_gaussian, 8);
		
		//展示结果
        imshow("img_src", img_src);
        imshow("img_damage", img_damage);
        imshow("img_mask", img_mask);
        imshow("img_result_near", img_result_near);
        imshow("img_result_bilinear", img_result_bilinear);
        imshow("img_result_rbf_mq", img_result_rbf_mq);
        imshow("img_result_rbf_tps", img_result_rbf_tps);
        imshow("img_result_rbf_g", img_result_rbf_g);
        waitKey(0);

  //      //保存mask
  //      imwrite(image_mask_dir + "scribbles/" + to_string(index) + "_mask.bmp", img_mask);
		////保存结果
  //      imwrite(image_reult_dir + "scribbles/" + to_string(index)+"_near.bmp", img_result_near);
  //      imwrite(image_reult_dir + "scribbles/" + to_string(index) + "_bilinear.bmp", img_result_bilinear);
  //      imwrite(image_reult_dir + "scribbles/" + to_string(index) + "_rbf.bmp", img_result_rbf);


        cout << "\n";
	}
    cout << "close images to continue\n\n";


    //task2 loss
    cout << "task2 fixed ratio loss" << endl;
    for (int index = 1; index <= 4; index++) {
        //读取原图
        Mat img_src = imread(image_src_dir + to_string(index) + ".bmp");
        vector<double> fixed_ratio={0.1,0.3,0.5,0.7,0.9};
    	//不同丢失率
    	for(int m=0;m<fixed_ratio.size();m++) {
            cout << "image " << to_string(index) << " lose "<<to_string(fixed_ratio[m]) << endl;
            Mat img_damage;
			Mat img_mask=deletePixel(img_src, img_damage, fixed_ratio[m]);//随机丢点

            //修复
            Mat img_result_near = interpolation_nearest(img_damage, img_mask, 0);//8邻域
            //Mat img_result_near_1 = interpolation_nearest(img_damage, img_mask, 1);//4邻域
            Mat img_result_bilinear = interpolation_bilinear(img_damage, img_mask, 0);//8邻域
            Mat img_result_rbf = interpolation_rbf(img_damage, img_mask, rbfphi_multiquadrics, 8);

            //展示结果
            imshow("img_src", img_src);
            imshow("img_damage", img_damage);
            imshow("img_mask", img_mask);
            imshow("img_result_near", img_result_near);
            imshow("img_result_bilinear", img_result_bilinear);
            imshow("img_result_rbf", img_result_rbf);
            waitKey(0);

            //保存mask
            imwrite(image_mask_dir + to_string(index) + "_mask.bmp", img_mask);
            imwrite(image_damage_dir + "loss/" + to_string(index) + "_" + to_string(fixed_ratio[m]) + "_loss.bmp", img_damage);

            //保存结果
            //imwrite(image_reult_dir + "scribbles/" + to_string(index)+"_near.bmp", img_result_near);
            //imwrite(image_reult_dir + "scribbles/" + to_string(index) + "_bilinear.bmp", img_result_bilinear);
            //imwrite(image_reult_dir + "scribbles/" + to_string(index) + "_bilinear.bmp", img_result_bilinear);

            cout << "\n";
    		
    	}

    }
    cout << "close images to continue\n\n";


    //task 2 
  //  for (int index = 1; index <= 4; index++) {
  //      cout << "image " << to_string(index) << " loss" << endl;
  //      
  //      Mat img_src = imread(image_src_dir + to_string(index) + ".bmp");//读取原图
  //      Mat img_damage;
		//Mat img_mask=deletePixel(img_src, img_damage, 0.80);//随机丢点
  //      imshow("img_src", img_src);
  //      imshow("img_damage", img_damage);
  //      imshow("img_mask", img_mask);
  //      //waitKey();
  //      Mat img_result_near = interpolation_nearest(img_damage, img_mask, 0);//8邻域
  //      Mat img_result_near_1 = interpolation_nearest(img_damage, img_mask, 1);//4邻域
  //      //Mat img_result_bilinear = interpolation_bilinear(img_damage, img_mask,0);
  //      //Mat img_result_bilinear_old = interpolation_bilinear_old(img_damage, img_mask);
  //      //Mat img_result_rbf = interpolation_rbf(img_damage, img_mask, 1, 10);
  //      imshow("img_result_near", img_result_near);
  //      imshow("img_result_near_1", img_result_near_1);
  //      //imshow("img_result_bilinear", img_result_bilinear);
  //      //imshow("img_result_bilinear_old", img_result_bilinear_old);
  //      //imshow("img_result_rbf", img_result_rbf);
  //      waitKey(0);
  //      //保存结果
  //  }

    
    //Mat img_damage = imread(imageDir + "damage/" + to_string(index) + ".bmp");
    //Mat img_mask = getScribbleMask(img_src, img_damage);
	
 //   Mat img_lost;
	//Mat img_lost_mask=deletePixel(img_src, img_lost, 0.50);//随机丢点

    //Mat img_result = interpolation_nearest(img_damage, img_mask, 1);
	//Mat img_result = interpolation_nearest(img_lost, img_lost_mask, 1);
    //Mat img_result = interpolation_bilinear(img_lost, img_lost_mask);
    //Mat img_result=interpolation_rbf(img_lost, img_lost_mask, 1,10);
	
    //Mat img_result = interpolation_nearest(img_lost, img_lost_mask, 1);
    
    //imshow("img_result", img_result);
    //imshow("img_src", img_src);
    //imshow("img_damage", img_damage);
    //imshow("img_mask", img_mask);
    //imshow("img_lost", img_lost);
    //imshow("img_lost_mask", img_lost_mask);
    //waitKey();

	
    //cout << metric_MSE(img_src, img_damage)<<endl;
    //cout << metric_PSNR(img_src, img_damage) << endl;
    //cout << metric_SSIM(img_src, img_damage) << endl;


    //imwrite(imageDir + "result\\" + to_string(index) + "_near_2.bmp", img_result);

}


//最近邻插值
Mat interpolation_nearest(const Mat& image, const Mat& mask_in, int flag) {
    cout << "nearest interpolating ..." << endl;
    if (image.size != mask_in.size) {
        cout << "size does not match" << endl;
        return image;
    }
    Mat result = image.clone();
    Mat mask = mask_in.clone();
    vector<pair<int, int>> lost;

    //统计所有待补点
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<uchar>(i, j) == 255) {
                lost.push_back(pair<int, int>(i, j));
            }
        }
    }

	//循环直至所有候补点全部被填充
    while (!lost.empty()) {
    	
    	//遍历所有待补点
        for(auto it=lost.begin();it<lost.end(); ) {
            vector<pair<int, int>> neighbor;

        	//遍历8邻域
        	for(int i=it->first-1;i<=it->first+1;i++) {
                for(int j=it->second-1;j<=it->second+1;j++) {
                	
                	//越界判断
                    if(i<0||i>=result.rows||j<0||j>=result.cols||(i==it->first&&j==it->second))
                        continue;
                	
                    //改为4邻域
                	if(flag==1) {
                		if((i!=it->first)&&(j!=it->second))
                            continue;
                	}

                	//将存在的邻居综合
                    if (mask.at<uchar>(i, j) != 255) {
                        neighbor.push_back(pair<int, int>(i, j));
                    }
                }
        	}

        	
            //判断邻域数量
            if (!neighbor.empty() ) {
                //随机选一个邻居填入
                int r = rand() % neighbor.size();//伪随机数(开销有点大)
            	
                result.at<Vec3b>(it->first, it->second) = result.at<Vec3b>(neighbor[r].first, neighbor[r].second);

                mask.at<uchar>(it->first, it->second) = 0;//标记为已填
                it = lost.erase(it);
            }
            else {
                it++;
            }

        }
    }
	

    return result;
}

//双线性插值
Mat interpolation_bilinear(const Mat& image, const Mat& mask_in, int flag) {
    cout << "bilinear interpolating ..." << endl;
    if (image.size != mask_in.size) {
        cout << "size does not match" << endl;
        return image;
    }
    Mat result = image.clone();
    Mat mask = mask_in.clone();
    vector<pair<int, int>> lost;

	//待补点
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<uchar>(i, j) == 255) {
                lost.push_back(pair<int, int>(i, j));
            }
        }
    }

    //循环直至所有候补点全部被填充
    while (!lost.empty()) {

        //遍历所有待补点
        for (auto it = lost.begin(); it < lost.end(); ) {

            vector<pair<int, int>> neighbor;
        	
            //遍历8邻域
            for (int i = it->first - 1; i <= it->first + 1; i++) {
                for (int j = it->second - 1; j <= it->second + 1; j++) {

                    //越界判断
                    if (i < 0 || i >= result.rows || j < 0 || j >= result.cols || (i == it->first && j == it->second))
                        continue;

                    //改为4邻域
                    if (flag == 1) {
                        if ((i != it->first) && (j != it->second))
                            continue;
                    }

                    if (mask.at<uchar>(i, j) != 255) {
                        neighbor.push_back(pair<int, int>(i, j));
                    }
                }
            }
            
        	//判断邻域数量
            int ne = neighbor.size();
        	if(neighbor.size()>=4) {
                //求平均
                vector<int> sum(3, 0);
                for(auto it2=neighbor.begin();it2<neighbor.end();++it2) {
                	for(int m=0;m<3;m++) {
                        sum[m] += result.at<Vec3b>(it2->first, it2->second)[m];
                	}
                }
                result.at<Vec3b>(it->first, it->second) = Vec3b(sum[0]/ne , sum[1]/ne, sum[2]/ne);
        		
                mask.at<uchar>(it->first, it->second) = 0;
                it = lost.erase(it);
        	}else {
                ++it;
        	}

        }
    }
	


    return result;
}

//rbf
Mat interpolation_rbf_new(const Mat& image, const Mat& mask_in, MatrixXd RBFFunction(MatrixXd r, double const_num),int neighbor_distance) {
    cout << "rbf interpolating ..." << endl;
	
    if (image.size != mask_in.size) {
        cout << "size does not match" << endl;
        return image;
    }
    Mat result = image.clone();
    Mat mask = mask_in.clone();
    vector<pair<int, int>> lost;

    //待补点
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<uchar>(i, j) == 255) {
                lost.push_back(pair<int, int>(i, j));
            }
        }
    }

    //循环直至所有候补点全部被填充
    while (!lost.empty()) {

        //遍历所有待补点
        int i, j;
        for (auto it = lost.begin(); it < lost.end(); ) {
            i = it->first;
            j = it->second;

            //遍历邻域
            int exist_count = 0;
            //统计邻域已知点数量
            for (int x = i - neighbor_distance; x < i + neighbor_distance; x++) {
                if (x < 0 || x >= result.rows)
                    continue;

                for (int y = j - neighbor_distance; y < j + neighbor_distance; y++) {
                    if (y < 0 || y >= result.cols)
                        continue;
                    if (mask.at<uchar>(x, y) != 255)
                        exist_count++;

                }
            }

            //没有已知的邻居
            if (exist_count == 0) {
                //cout << "neighbor distance is too small" << endl;
                ++it;
                continue;
            }


            //分配矩阵
            MatrixXd exist_pos(exist_count, 2);
            MatrixXd exist_b_value(exist_count, 1);
            MatrixXd exist_g_value(exist_count, 1);
            MatrixXd exist_r_value(exist_count, 1);

            vector<MatrixXd> exist_value;
            exist_value.push_back(exist_b_value);
            exist_value.push_back(exist_g_value);
            exist_value.push_back(exist_r_value);

            MatrixXd pre_pos(1, 2);
            MatrixXd pre_value;

            pre_pos << i, j;

            int index_exist = 0;

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

                }
            }

            //计算未知值
            for (int index = 0; index < 3; index++) {
                MatrixXd rbfcoeff = rbfcreate(exist_pos, exist_value[index], RBFFunction, 0.4444, 0);
                pre_value = rbfinterp(exist_pos, rbfcoeff, pre_pos, RBFFunction, 0.4444);
                result.at<Vec3b>(i, j)[index] = (int)(pre_value(0, 0));
            }

            //标记该点为已填
            mask.at<uchar>(it->first, it->second) = 0;
            it = lost.erase(it);

        }
    }

    return result;
}


Mat interpolation_rbf(const Mat& image, const Mat& mask, MatrixXd RBFFunction(MatrixXd r, double const_num), int neighbor_distance) {
    cout << "rbf interpolating ..." << endl;

    if (image.size != mask.size) {
        cout << "size does not match" << endl;
        return image;
    }
    Mat result = image.clone();


    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            //判断是否需要插值
            if (mask.at<uchar>(i, j) == 255) {
                //遍历邻域

                int exist_count = 0;        	

                //统计数量
                for (int x = i - neighbor_distance; x < i + neighbor_distance; x++) {
                    if (x < 0 || x >= result.rows)
                        continue;

                    for (int y = j - neighbor_distance; y < j + neighbor_distance; y++) {
                        if (y < 0 || y >= result.cols)
                            continue;
                        if (mask.at<uchar>(x, y) != 255)
                            exist_count++;

                    }
                }

                //可参考的数量过少
                if (exist_count == 0) {
                    cout << "neighbor distance is too small" << endl;
                    continue;
                }

                //分配矩阵
                MatrixXd exist_pos(exist_count, 2);
                MatrixXd exist_b_value(exist_count, 1);
                MatrixXd exist_g_value(exist_count, 1);
                MatrixXd exist_r_value(exist_count, 1);

                vector<MatrixXd> exist_value;
                exist_value.push_back(exist_b_value);
                exist_value.push_back(exist_g_value);
                exist_value.push_back(exist_r_value);

                MatrixXd pre_pos(1, 2);
                MatrixXd pre_value;

                pre_pos << i, j;

                int index_exist = 0;

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

                    }
                }

                //计算未知值
                //cout << "solve" << endl;
                for (int index = 0; index < 3; index++) {
                    MatrixXd rbfcoeff = rbfcreate(exist_pos, exist_value[index], RBFFunction, 0.4444, 0);
                    pre_value = rbfinterp(exist_pos, rbfcoeff, pre_pos, RBFFunction, 0.4444);
                    result.at<Vec3b>(i, j)[index] = (int)(pre_value(0, 0));
                }

            }
        }
    }

    return result;
}


Mat deletePixel(const Mat& imageSrc, Mat& imageDst, double ratio) {
	
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
    subtract(original, processed, temp, noArray(), CV_64FC3);
    multiply(temp, temp, temp, 1, CV_64FC3);
    Scalar temps = mean(temp);
    return (temps.val[0] + temps.val[1] + temps.val[2]) / 3.0;
}

double metric_PSNR(const Mat& original, const Mat& processed) {
    if (original.size() != processed.size()) {
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
