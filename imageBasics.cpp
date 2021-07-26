#include <iostream>
#include <chrono>
using namespace std;

#include <stdio.h>
#include <sys/time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iomanip>
#include <fstream>

#define PI 3.1415926

///////512×512鱼眼影像上任一像素(i,j)对应1920×1920影像上的点的坐标(ccols,crows)
struct fisheye
{
    int i, j;
    float crow, ccol; ////height,width
};

struct RGB{
    double b,g,r;//////三通道RGB影像
};

////双线性插值，i,j带小数，通过取整，双线性内插获得结果，返回值是double类型的(i,j)处的b,g,r
RGB bilinear_Interpolation(cv::Mat correctedImage, double i, double j){

    double i_float = i - (int)i;
    double j_float = j - (int)j;/////i,j的小数部分
    int izheng=(int)i;
    int jzheng=(int)j;/////i,j的整数部分

    int height_=correctedImage.rows;
    //int width_=correctedImage.cols;/////影像的宽和高
    int width_=1920;
    unsigned char *pImage = correctedImage.data;

    double b=0.0,g=0.0,r=0.0;///RGB三通道
    ////////对每个通道都进行双线性插值

    int value00 = pImage[(izheng * width_ + jzheng) * 3 + 0];
    int value10 = pImage[((izheng + 1) * width_ + jzheng) * 3 + 0];
    int value11 = pImage[((izheng + 1) * width_ + jzheng + 1) * 3 + 0];
    int value01 = pImage[(izheng * width_ + jzheng + 1) * 3 + 0];
    b=(value00 * (1 - i_float) * (1 - j_float) + value10 * i_float * (1 - j_float) + value11 * i_float * j_float + value01 * (1 - i_float) * j_float);
    
    value00 = pImage[(izheng * width_ + jzheng) * 3 + 1];
    value10 = pImage[((izheng + 1) * width_ + jzheng) * 3 + 1];
    value11 = pImage[((izheng + 1) * width_ + jzheng + 1) * 3 + 1];
    value01 = pImage[(izheng * width_ + jzheng + 1) * 3 + 1];
    g=(value00 * (1 - i_float) * (1 - j_float) + value10 * i_float * (1 - j_float) + value11 * i_float * j_float + value01 * (1 - i_float) * j_float);
    
    value00 = pImage[(izheng * width_ + jzheng) * 3 + 2];
    value10 = pImage[((izheng + 1) * width_ + jzheng) * 3 + 2];
    value11 = pImage[((izheng + 1) * width_ + jzheng + 1) * 3 + 2];
    value01 = pImage[(izheng * width_ + jzheng + 1) * 3 + 2];
    r=(value00 * (1 - i_float) * (1 - j_float) + value10 * i_float * (1 - j_float) + value11 * i_float * j_float + value01 * (1 - i_float) * j_float);

    RGB rgbpoint;
    rgbpoint.b = b;
    rgbpoint.g = g;
    rgbpoint.r = r;

    //int value00 = correctedImage.at<uchar>((int)i, (int)j);
    //int value10 = correctedImage.at<uchar>((int)i + 1, (int)j);
    //int value11 = correctedImage.at<uchar>((int)i + 1, (int)j + 1);
    //int value01 = correctedImage.at<uchar>((int)i, (int)j + 1);

    //return (value00 * (1 - i_float) * (1 - j_float) + value10 * i_float * (1 - j_float) + value11 * i_float * j_float + value01 * (1 - i_float) * j_float);
    return rgbpoint;
}

//////512×512的鱼眼影像上任一点(i,j)对应的1920×1920影像上的点的坐标(crow,ccol),存储在fisheye*类型矩阵result中

int fisheyecount=0;///计数器，全局变量，记录鱼眼影像中处理点的个数
fisheye *GetPosRelationFishAndOpen(cv::Size openImage, cv::Size fisheyeImage){
    int rows = openImage.height;
    int cols = openImage.width;
    int fisheyeImageRows = fisheyeImage.height;
    int fisheyeImageCols = fisheyeImage.width;

    float X, Y;
    float R, Sita2D, Omga2D;
    float x, y, z;
    float r, Sita3D, Omga3D;
    int crow, ccol;
    fisheye *result = new fisheye[fisheyeImageRows * fisheyeImageCols];

    /////////////////遍历512×512鱼眼影像上每一个点
    for (int i = 0; i < fisheyeImageRows; i++)
    {
        for (int j = 0; j < fisheyeImageCols; j++)
        {
            //判断是否在圆内
            if (((i - fisheyeImageRows / 2) * (i - fisheyeImageRows / 2) + (j - fisheyeImageCols / 2) * (j - fisheyeImageCols / 2)) < (fisheyeImageRows * fisheyeImageRows / 4))
            {
                //计算对应的球面坐标系坐标
                X = j - fisheyeImageCols / 2;
                Y = i - fisheyeImageCols / 2;
                if (X > 0)
                {
                    Sita2D = atan(Y / X);
                }
                else if (X < 0)
                {
                    Sita2D = PI + atan(Y / X);
                }
                else if (Y > 0)
                {
                    Sita2D = PI / 2;
                }
                else
                {
                    Sita2D = 3 * PI / 2;
                }
                Omga2D = sqrt(X * X + Y * Y) / (fisheyeImageCols / PI);

                r = 2 * fisheyeImageCols / PI;

                x = r * sin(Omga2D) * cos(Sita2D);
                y = r * sin(Omga2D) * sin(Sita2D);
                z = r * cos(Omga2D);

                //计算球面角度坐标
                if (x > 0)
                {
                    Sita3D = PI / 2 + asin(x / sqrt(x * x + z * z));
                }
                else
                {
                    Sita3D = PI / 2 - asin(abs(x) / sqrt(x * x + z * z));
                }
                Omga3D = asin(y / r);
                //双线性插值(Sita,Omga)
                //crow = (Omga3D + PI / 2) / PI * (rows - 2);
                //ccol = (Sita3D) / PI * (cols - 2);
                crow = (Omga3D + PI / 2) / PI * rows - 1;
                ccol = (Sita3D) / PI * cols - 1;

                ////将结果记录并存入result中
                fisheye fisheyepoint;
                fisheyepoint.i = i;
                fisheyepoint.j = j;
                fisheyepoint.crow = crow;
                fisheyepoint.ccol = ccol;

                result[fisheyecount] = fisheyepoint;
                fisheyecount++;
            } /////计算一个对应点坐标完成
        }
    } /////计算所有对应点坐标完成

    return result;
}


int main(int argc, char **argv){

    //1.初始化对应map，求得512×512鱼眼影像和1920×1920影像各像素对应关系
    //512×512鱼眼影像上任一像素(i,j)对应1920×1920影像上的点的坐标
    fisheye *fisheye512 = GetPosRelationFishAndOpen(cv::Size(1920, 1920), cv::Size(512, 512));//////存储对应关系
   

    /******** 获取视频文件(实例化的同时进行初始化)*******/
    cv::VideoCapture capture("qing1.MP4");

    if (!capture.isOpened())
    {
        return -1;
    }

    /********** 获取视频总帧数并打印*****************/
    long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
    cout << "total frames: " << totalFrameNumber << endl;

    cv::Mat frame;         //定义一个Mat变量，用来存放存储每一帧图像
    bool flags = true;     //循环标志位
    long currentFrame = 1; //定义当前帧

    while (flags)
    {
        capture.read(frame); // 读取视频每一帧,帧命名为frame
        stringstream str;    //stringstream字符串流，将long类型的转换成字符型传给对象str

        struct timeval stamp; ///// 在Linux系统下获得当前时间戳
        gettimeofday(&stamp, NULL);

        

        //cout << "正在处理第" << currentFrame << "帧" << endl;

        /////打开记录抽取的帧的时间戳的txt文件并写入
        ofstream timefile("timestamp.txt", ios::app);
        if (!timefile)
        {
            cout << "Unable to open otfile";
            exit(1);
        }

        /***设置每10帧获取一次帧***/
        if (currentFrame % 3 == 1)
        {
            /////命名，考虑时间的小数部分不是6位的情况，要补0
            if (stamp.tv_usec >= 100000)
            {
                str << stamp.tv_sec << stamp.tv_usec << ".png";    /////将每一帧以时间戳命名
                timefile << stamp.tv_sec << stamp.tv_usec << endl; //////将输出的帧对应的时间戳写入txt文件
            }
            if (stamp.tv_usec < 100000 && stamp.tv_usec >= 10000)
            {
                str << stamp.tv_sec << "0" << stamp.tv_usec << ".png";    /////将每一帧以时间戳命名
                timefile << stamp.tv_sec << "0" << stamp.tv_usec << endl; //////将输出的帧对应的时间戳写入txt文件
            }
            if (stamp.tv_usec < 10000 && stamp.tv_usec >= 1000)
            {
                str << stamp.tv_sec << "00" << stamp.tv_usec << ".png";    /////将每一帧以时间戳命名
                timefile << stamp.tv_sec << "00" << stamp.tv_usec << endl; //////将输出的帧对应的时间戳写入txt文件
            }
            if (stamp.tv_usec < 1000 && stamp.tv_usec >= 100)
            {
                str << stamp.tv_sec << "000" << stamp.tv_usec << ".png";    /////将每一帧以时间戳命名
                timefile << stamp.tv_sec << "000" << stamp.tv_usec << endl; //////将输出的帧对应的时间戳写入txt文件
            }
            if (stamp.tv_usec < 100 && stamp.tv_usec >= 10)
            {
                str << stamp.tv_sec << "0000" << stamp.tv_usec << ".png";    /////将每一帧以时间戳命名
                timefile << stamp.tv_sec << "0000" << stamp.tv_usec << endl; //////将输出的帧对应的时间戳写入txt文件
            }
            if (stamp.tv_usec < 10 && stamp.tv_usec >= 1)
            {
                str << stamp.tv_sec << "00000" << stamp.tv_usec << ".png";    /////将每一帧以时间戳命名
                timefile << stamp.tv_sec << "00000" << stamp.tv_usec << endl; //////将输出的帧对应的时间戳写入txt文件
            }
           
            cv::imwrite("./images/" + str.str(), frame); // 将帧转成图片输出

            cv::Mat imageOrigin = frame.clone(); /////读取3840×1920的全景影像

          
            //////裁剪得到4个方向1920×1920的影像
             ////////裁剪影像
            unsigned char *pImageOrigin=imageOrigin.data; ////指向要裁剪的帧的像素灰度值数据
            
            //////左裁剪影像
            cv::Mat imageOriginLeft;
            imageOriginLeft.create(imageOrigin.rows, imageOrigin.cols / 2,CV_8UC3); /////构造1920*1920的图片
            unsigned char *pImageLeft=imageOriginLeft.data;
            int height=imageOriginLeft.rows;///1920
            int width=imageOriginLeft.cols;///1920
           
            //前裁剪影像
            cv::Mat imageOriginFront;
            imageOriginFront.create(height, width,CV_8UC3); /////构造1920*1920的图片
            unsigned char *pImageFront=imageOriginFront.data;

            //右裁剪影像
            cv::Mat imageOriginRight;
            imageOriginRight.create(height, width,CV_8UC3); /////构造1920*1920的图片
            unsigned char *pImageRight=imageOriginRight.data;

            //////将原影像帧一般的数据赋给裁剪影像
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    pImageLeft[(i * width + j) * 3 + 0] = pImageOrigin[(i * width*2 + j) * 3 + 0];////左裁剪影像
                    pImageLeft[(i * width + j) * 3 + 1] = pImageOrigin[(i * width*2 + j) * 3 + 1];
                    pImageLeft[(i * width + j) * 3 + 2] = pImageOrigin[(i * width*2 + j) * 3 + 2];

                    pImageFront[(i * width + j) * 3 + 0] = pImageOrigin[(i * width*2 + j+width/2) * 3 + 0];////前裁剪影像
                    pImageFront[(i * width + j) * 3 + 1] = pImageOrigin[(i * width*2 + j+width/2) * 3 + 1];
                    pImageFront[(i * width + j) * 3 + 2] = pImageOrigin[(i * width*2 + j+width/2) * 3 + 2];
                    
                    pImageRight[(i * width + j) * 3 + 0] = pImageOrigin[(i * width*2 + j+width) * 3 + 0];////右裁剪影像
                    pImageRight[(i * width + j) * 3 + 1] = pImageOrigin[(i * width*2 + j+width) * 3 + 1];
                    pImageRight[(i * width + j) * 3 + 2] = pImageOrigin[(i * width*2 + j+width) * 3 + 2];


                }
            }           
            
            //后裁剪影像
            cv::Mat imageOriginBack;
            imageOriginBack.create(imageOrigin.rows, imageOrigin.cols / 2,CV_8UC3); /////构造1920*1920的图片
            unsigned char *pImageBack=imageOriginBack.data;

            //////裁剪imageOrigin(2880,0)-(3840,1920)和(0,0)-(960,1920)部分的影像
            for (int j = 0; j < height; j++) /////height(0,1920)
            {
                for (int k = 0; k < width; k++) ////width(0,1920)
                {
                    if (k < width / 2) ////height<960
                    {
                        pImageBack[(j * width + k) * 3 + 0] = pImageOrigin[(j * imageOrigin.cols + k + 3 * width / 2) * 3 + 0]; /////2880<width<3840
                        pImageBack[(j * width + k) * 3 + 1] = pImageOrigin[(j * imageOrigin.cols + k + 3 * width / 2) * 3 + 1];
                        pImageBack[(j * width + k) * 3 + 2] = pImageOrigin[(j * imageOrigin.cols + k + 3 * width / 2) * 3 + 2];
                    }
                    else
                    {
                        pImageBack[(j * width + k) * 3 + 0] = pImageOrigin[(j * imageOrigin.cols + k - width / 2) * 3 + 0];
                        pImageBack[(j * width + k) * 3 + 1] = pImageOrigin[(j * imageOrigin.cols + k - width / 2) * 3 + 1];
                        pImageBack[(j * width + k) * 3 + 2] = pImageOrigin[(j * imageOrigin.cols + k - width / 2) * 3 + 2]; /////0<width<960
                    }
                }
            }

            ////////3.根据512×512的鱼眼影像上任一点(i,j)对应的1920×1920影像上的点坐标(crows,cols)之间的对应关系，求得512*512影像上各个像素的灰度值

  /////////////4个方向鱼眼影像初始化////////////////////////////////////
            cv::Mat outImageFront;
            outImageFront.create(512, 512,CV_8UC3);
            cv::Mat outImageBack;
            outImageBack.create(512, 512,CV_8UC3);
            cv::Mat outImageRight;
            outImageRight.create(512, 512,CV_8UC3);
            cv::Mat outImageLeft;
            outImageLeft.create(512, 512,CV_8UC3);

            unsigned char *pOutBack=outImageBack.data;
            unsigned char *pOutFront=outImageFront.data;
            unsigned char *pOutLeft=outImageLeft.data;
            unsigned char *pOutRight=outImageRight.data;

            for(int i=0;i<512*512*3;i++){/////初始化，影像全为黑色
                pOutBack[i]=0;
                pOutFront[i]=0;
                pOutLeft[i]=0;
                pOutRight[i]=0;
            }

            for (int m = 0; m < fisheyecount; m++) //////遍历该影像上所有点
            {
                fisheye fisheyeImage = fisheye512[m]; ////读取鱼眼影像上的一个点

                int image_i = fisheyeImage.i;
                int image_j = fisheyeImage.j; ////读取鱼眼影像对应的点的位置
                double image_crow = fisheyeImage.crow;
                double image_ccol = fisheyeImage.ccol; ////读取该点对应的1920×1920影像上的位置

                ////双线性插值求得double类型的rgb
                RGB rgbpointFront = bilinear_Interpolation(imageOriginFront, image_crow, image_ccol);
                pOutFront[(image_i * 512 + image_j) * 3 + 0] = (int)(rgbpointFront.b + 0.5);
                pOutFront[(image_i * 512 + image_j) * 3 + 1] = (int)(rgbpointFront.g + 0.5);
                pOutFront[(image_i * 512 + image_j) * 3 + 2] = (int)(rgbpointFront.r + 0.5);

                RGB rgbpointBack = bilinear_Interpolation(imageOriginBack, image_crow, image_ccol);
                pOutBack[(image_i * 512 + image_j) * 3 + 0] = (int)(rgbpointBack.b + 0.5);
                pOutBack[(image_i * 512 + image_j) * 3 + 1] = (int)(rgbpointBack.g + 0.5);
                pOutBack[(image_i * 512 + image_j) * 3 + 2] = (int)(rgbpointBack.r + 0.5);

                RGB rgbpointLeft = bilinear_Interpolation(imageOriginLeft, image_crow, image_ccol);
                pOutLeft[(image_i * 512 + image_j) * 3 + 0] = (int)(rgbpointLeft.b + 0.5);
                pOutLeft[(image_i * 512 + image_j) * 3 + 1] = (int)(rgbpointLeft.g + 0.5);
                pOutLeft[(image_i * 512 + image_j) * 3 + 2] = (int)(rgbpointLeft.r + 0.5);

                RGB rgbpointRight = bilinear_Interpolation(imageOriginRight, image_crow, image_ccol);
                pOutRight[(image_i * 512 + image_j) * 3 + 0] = (int)(rgbpointRight.b + 0.5);
                pOutRight[(image_i * 512 + image_j) * 3 + 1] = (int)(rgbpointRight.g + 0.5);
                pOutRight[(image_i * 512 + image_j) * 3 + 2] = (int)(rgbpointRight.r + 0.5);

            } //////遍历该影像上所有点结束

            cv::imwrite("./leftbefore/" + str.str(), imageOriginLeft);   // 将帧转成图片输出
            cv::imwrite("./rightbefore/" + str.str(), imageOriginRight); // 将帧转成图片输出
            cv::imwrite("./frontbefore/" + str.str(), imageOriginFront); // 将帧转成图片输出
            cv::imwrite("./backbefore/" + str.str(), imageOriginBack);   // 将帧转成图片输出

            //cv::imwrite("./left/" + str.str(), outImageLeft);   // 将帧转成图片输出
            //cv::imwrite("./right/" + str.str(), outImageRight); // 将帧转成图片输出
            cv::imwrite("./front/" + str.str(), outImageFront); // 将帧转成图片输出
            //cv::imwrite("./back/" + str.str(), outImageBack);   // 将帧转成图片输出

        } ////当前帧处理完成

        /**** 结束条件,当前帧数大于总帧数时候时，循环停止****/
        if (currentFrame >= totalFrameNumber)
        {
            flags = false;
            timefile.close(); ///关闭时间戳txt文件
            cout << "处理完毕" << endl;
        }

        currentFrame++;
    }

    cv::waitKey(0);
    return 0;
}

