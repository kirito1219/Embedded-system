#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <linux/fb.h>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <string>

using namespace std;
using namespace cv;
using namespace face;

/** Global variables **/
int64 t;
const int frame_width = 320;
const int frame_height  = 240;
const int frame_rate = 30;
/** Detect Model **/

String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "./haarcascades/haarcascade_eye_tree_eyeglasses.xml";
String face_model = "./train/trainner/trainner.yml";

CascadeClassifier face_cascade; // Declare the face classifier
CascadeClassifier eyes_cascade; // Declare the eyes classifier
Ptr<LBPHFaceRecognizer> model = Algorithm::load<LBPHFaceRecognizer>(face_model);//parameter Model

/** Function for face detection **/
void Detect(Mat& frame);

struct framebuffer_info
{
    uint32_t bits_per_pixel;    // depth of framebuffer
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
};

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path );

int main ( int argc, const char *argv[] )
{
    // variable to store the frame get from video stream
    cv::Mat frame;
    // open video stream device
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a5d5f5dacb77bbebdcbfb341e3d4355c1
    cv::VideoCapture camera(2);
	camera.set(CV_CAP_PROP_FRAME_WIDTH,frame_width);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT,frame_height);
    camera.set(CV_CAP_PROP_FPS,frame_rate);

    // get info of the framebuffer
    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    // open the framebuffer device
    // http://www.cplusplus.com/reference/fstream/ofstream/ofstream/
    ofstream ofs("/dev/fb0");
    // check if video stream device is opened success or not
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a9d2ca36789e7fcfe7a7be3b328038585
    if(!camera.isOpened()) {
        std::cerr << "Could not open video device." << std::endl;
    } 
    // set propety of the frame
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a8c6d8c2d37505b5ca61ffd4bb54e9a7c
    // https://docs.opencv.org/3.4.7/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
	
	// load detect  Model
	if(!face_cascade.load(face_cascade_name))
	 	std::cerr << "Error load face cascade" << endl;

	if(!eyes_cascade.load(eyes_cascade_name))
	 	std::cerr << "Error load eyes cascade" << endl;
	
	if(model->empty())
	 	std::cerr << "face model load fault" << endl;

	string str_ready;
	cout << "Input string (ready) to  Detect Face" << endl;
	cin >> str_ready;
	if(str_ready == "ready")
	{
	 	cout << "Start Face Detect" << endl;
     	while(true)
     	{
         	// get video frame from stream
         	// https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1
         	// https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#a199844fb74226a28b3ce3a39d1ff6765
     	 	t = getTickCount();
	 	 	camera >> frame;

	 	 	Detect(frame);
            resize(frame, frame, Size(640,480), 1, 1, INTER_LINEAR);
         	cv::Mat framebuffer_compat;

         	// get size of the video frame
         	// https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a146f8e8dda07d1365a575ab83d9828d1
         	cv::Size2f frame_size = frame.size();

         	// transfer color space from BGR to BGR565 (16-bit image) to fit the requirement of the LCD
         	// https://docs.opencv.org/3.4.7/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
         	// https://docs.opencv.org/3.4.7/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0
	 	 	cv::cvtColor(frame, framebuffer_compat, cv::COLOR_BGR2BGR565);

	 	 	cout << "Detect time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;
	 	 	// output the video frame to framebufer row by row
         	for ( int y = 0; y < frame_size.height; y++ )
         	{
             	// move to the next written position of output device framebuffer by "std::ostream::seekp()"
             	// http://www.cplusplus.com/reference/ostream/ostream/seekp/
	     	 	ofs.seekp(y*fb_info.xres_virtual*2);
	     	 	ofs.write(reinterpret_cast<char*>(framebuffer_compat.ptr(y)),frame_size.width*2);
             	// write to the framebuffer by "std::ostream::write()"
             	// you could use "cv::Mat::ptr()" to get the pointer of the corresponding row.
             	// you also need to cacluate how many bytes required to write to the buffer
             	// http://www.cplusplus.com/reference/ostream/ostream/write/
             	// https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a13acd320291229615ef15f96ff1ff738
         	}
     	}
     	// closing video stream
     	// https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#afb4ab689e553ba2c8f0fec41b9344ae6
	}
    camera.release();
    return 0;
}

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path )
{
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.
    int fd = -1;
    fd = open(framebuffer_device_path, O_RDWR);
    if (fd >= 0) {
        if (!ioctl(fd, FBIOGET_VSCREENINFO, &screen_info)) {
     	 	// put the required attributes in variable "fb_info" you found with "ioctl() and return it."
            fb_info.xres_virtual = screen_info.xres_virtual;
            fb_info.bits_per_pixel = screen_info.bits_per_pixel;
        }
    }
    // open deive with linux system call "open( )"
    // https://man7.org/linux/man-pages/man2/open.2.html

    // get attributes of the framebuffer device thorugh linux system call "ioctl()"
    // the command you would need is "FBIOGET_VSCREENINFO"
    // https://man7.org/linux/man-pages/man2/ioctl.2.html
    // https://www.kernel.org/doc/Documentation/fb/api.txt
    return fb_info;
};

void Detect(Mat& frame){
    /* Declare vector for faces and eyes */
    std::vector<Rect> faces, eyes;
    Mat frame_gray, frame_resize;
    int radius;
    /* Convert to gray scale */
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    /* Resize the grayscale Image */
    resize(frame_gray, frame_resize, Size(300,300), 1, 1, INTER_LINEAR);
    /* Histogram equalization */
    equalizeHist(frame_gray, frame_gray);
    /* Detect faces of different sizes using cascade classifier */
    face_cascade.detectMultiScale(frame_gray, faces, 1.2, 5, CASCADE_SCALE_IMAGE, Size(30, 30)); // 1.1 - > 1.00

    /** Draw circles around the faces **/
    for (size_t i = 0; i < faces.size(); i++)
    {
     	String name;
        Point center;
        /* Draw rectangular on face */
        rectangle(frame, faces[i], Scalar(255, 0, 0), 1.5, 8, 0);
        int label;
	 	double confidence = 0;
        Mat faceROI = frame_gray(faces[i]);
	 	model->predict(faceROI, label, confidence); // predict myself model

	 	Point LB(faces[i].x, faces[i].y);
	 	if(confidence < 90){
	 	 	if(label == 1){
	 	 	 	name = "DE YU HONG";
	 	 	 	putText(frame, name, LB, FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255));
	 	 	 	cout << "DE YU HONG" << endl;
	 	 	}
	 	 	else{
	  	 	 	name = "LI SIN JHANG";
	 	 	 	putText(frame, name, LB, FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255));
	 	 	 	cout << "LI SIN JHANG";
	 	 	}
	 	}
	 	else
	 	{
	 	 	name = "Unknown";
	 	 	putText(frame, name, LB, FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255));
	 	 	cout << "Unknown" << endl;
	 	}
	 	cout << confidence << endl; // Debug for modify threshhold
        /* Detection of eyes int the input image */
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 1, CASCADE_SCALE_IMAGE, Size(3, 3)); 
        /** Draw circles around eyes **/
        for (size_t j = 0; j < eyes.size(); j++) 
        {
            center.x = cvRound((faces[i].x + eyes[j].x + eyes[j].width*0.5));
       	    center.y = cvRound((faces[i].y + eyes[j].y + eyes[j].height*0.5));
            radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
            circle(frame, center, radius, Scalar(0, 255, 0), 3, 8, 0);
        }
    }
    // Show Processed Image with detected faces
}
