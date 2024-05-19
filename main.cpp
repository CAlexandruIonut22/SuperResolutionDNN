#include <iostream>
#include <opencv2/opencv_modules.hpp>
#include <cstdio>
#include <Windows.h>
#include <filesystem>

// Add Plot library

#ifdef HAVE_OPENCV_QUALITY
#include <opencv2/dnn_superres.hpp>
#include <opencv2/quality.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;
using namespace dnn_superres;

static void showBenchmark(vector<Mat> images, string title, Size imageSize,
    const vector<String> imageTitles,
    const vector<double> psnrValues,
    const vector<double> ssimValues)
{
    int fontFace = FONT_HERSHEY_PLAIN;
    int fontScale = 1;
    Scalar fontColor = Scalar(0, 128, 0);

    int len = static_cast<int>(images.size());

    int cols = 2, rows = 2;

    Mat fullImage = Mat::zeros(Size((cols * 10) + imageSize.width * cols, (rows * 10) + imageSize.height * rows),
        images[0].type());

    stringstream ss;
    int h_ = -1;
    for (int i = 0; i < len; i++) {

        int fontStart = 17;
        int w_ = i % cols;
        if (i % cols == 0)
            h_++;

        Rect ROI((w_ * (10 + imageSize.width)), (h_ * (10 + imageSize.height)), imageSize.width, imageSize.height);
        Mat tmp;
        resize(images[i], tmp, Size(ROI.width, ROI.height));

        ss << imageTitles[i];
        putText(tmp,
            ss.str(),
            Point(5, fontStart),
            fontFace,
            fontScale,
            fontColor,
            1,
            16);

        ss.str("");
        fontStart += 20;

        ss << "PSNR: " << psnrValues[i];
        putText(tmp,
            ss.str(),
            Point(5, fontStart),
            fontFace,
            fontScale,
            fontColor,
            1,
            16);

        ss.str("");
        fontStart += 20;

        ss << "SSIM: " << ssimValues[i];
        putText(tmp,
            ss.str(),
            Point(5, fontStart),
            fontFace,
            fontScale,
            fontColor,
            1,
            16);

        ss.str("");
        fontStart += 20;

        tmp.copyTo(fullImage(ROI));
    }

    namedWindow(title, 1);
    imshow(title, fullImage);
    waitKey();
}

static Vec2d getQualityValues(Mat orig, Mat upsampled)  // find another way to calculate psnr + ssim
{
    double psnr = PSNR(upsampled, orig);
    Scalar q = quality::QualitySSIM::compute(upsampled, orig, noArray());
    double ssim = mean(Vec3d((q[0]), q[1], q[2]))[0];
    return Vec2d(psnr, ssim);
}


int main(int argc, char* argv[])
{
    // TODO: CHANGE INPUT FORM - > to wait for values and different questions
    if (argc < 4) {
        cout << "The image path | Path to image" << endl;
        cout << "\t The algorithm | edsr, espcn, fsrcnn or lapsrn" << endl; 
        cout << "\t The path to the model file 2 \n"; 
        return -1;
    }


    string path = string(argv[1]);
    string algorithm = string(argv[2]);
    for (auto& x : algorithm) {
        x = tolower(x);
    }

    string model = string(argv[3]);
    int scale = int(model[model.length() - 4] - '0'); // Takes the scale number from the file name, 
    // keep in mind file format for models matter

    Mat img = imread(path);
    if (img.empty()) {
        cerr << "Couldn't load image: " << img << "\n";
        return -2;
    }
    
    //Get image dimensions
    int width = img.cols - (img.cols % scale);
    int height = img.rows - (img.rows % scale);
    // Create cropped image
    Mat cropped = img(Rect(0, 0, width, height));

    Mat img_downscaled;
    // Create downscaled image
    resize(cropped, img_downscaled, Size(), 1.0 / scale, 1.0 / scale);

    // SR using DNN method implementation
    DnnSuperResImpl sr;

    vector <Mat> allImages;
    Mat img_new;
    Mat img_new2;

    sr.readModel(model);
    sr.setModel(algorithm, scale);
    sr.upsample(img_downscaled, img_new);
    sr.upsample(cropped, img_new2);

    // Show first img
    imshow("Final img", img_new);
    // Show second img
    imshow("Final img2", img_new2);


    // Create vectors for values of PSNR and SSIM
    vector<double> psnrValues = vector<double>();
    vector<double> ssimValues = vector<double>();

    // Create output file stream to write values to plot afterwards.
    /*
    ofstream dataFile("plt_data.txt");
    if (!dataFile.is_open()) {
        cerr << "Error:Unable to open data file!!!" << endl;
        return -1;
    }
    */  

    // Call function getQualityValues to get the PSNR and SSIM values of img_new
    Vec2f quality = getQualityValues(cropped, img_new);

    psnrValues.push_back(quality[0]);
    //dataFile << quality[0]; // write data to file
    ssimValues.push_back(quality[1]);
    // Print values of PSNR and SSIM
    cout << sr.getAlgorithm() << ":" << endl;
    cout << "PSNR: " << quality[0] << " SSIM: " << quality[1] << endl;
    cout << "----------------------" << endl;
    // SR with DNN method end here

    // Classic methods start here
    // 
    //BICUBIC

    Mat bicubic;
    resize(img_downscaled, bicubic, Size(), scale, scale, INTER_CUBIC);
    quality = getQualityValues(cropped, bicubic);

    psnrValues.push_back(quality[0]);
    //dataFile << quality[0]; // write data to file
    ssimValues.push_back(quality[1]);

    cout << "Bicubic " << endl;
    cout << "PSNR: " << quality[0] << " SSIM: " << quality[1] << endl;
    cout << "----------------------" << endl;

    //applyBicubic(img_downscaled, scale, cropped, psnrValues, ssimValues); // call function to apply Bicubic algorithm

    //NEAREST NEIGHBOR
    Mat nearest;
    resize(img_downscaled, nearest, Size(), scale, scale, INTER_NEAREST);
    quality = getQualityValues(cropped, nearest);

    psnrValues.push_back(quality[0]);
    //dataFile << quality[0]; // write data to file
    ssimValues.push_back(quality[1]);

    cout << "Nearest neighbor" << endl;
    cout << "PSNR: " << quality[0] << " SSIM: " << quality[1] << endl;
    cout << "----------------------" << endl;

    //LANCZOS
    Mat lanczos;
    resize(img_downscaled, lanczos, Size(), scale, scale, INTER_LANCZOS4);
    quality = getQualityValues(cropped, lanczos);

    psnrValues.push_back(quality[0]);
    //dataFile << quality[0]; // write data to file
    ssimValues.push_back(quality[1]);

    cout << "Lanczos" << endl;
    cout << "PSNR: " << quality[0] << " SSIM: " << quality[1] << endl;
    cout << "-----------------------------------------------" << endl;

    vector <Mat> imgs{ img_new, bicubic, nearest, lanczos };
    vector <String> titles{ sr.getAlgorithm(), "Bicubic", "Nearest neighbor", "Lanczos" };

    //dataFile.close();
    showBenchmark(imgs, "Quality benchmark", Size(bicubic.cols, bicubic.rows), titles, psnrValues, ssimValues);

    // TODO : Plot for PSNR values and SSIM values
        
    waitKey(0); 
 
    //}

    return 0;
}
 

#else
int main()
{
    // If OpenCV lib issue, display the following: 
    std::cout << "This sample requires the OpenCV Quality module." << std::endl;
    return 0;
}
#endif