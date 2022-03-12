#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::face;

/** Function for face detection **/
void DetectAndDraw(Mat frame);

/** Global variables **/
String window_name = "Face detection";
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "./haarcascades/haarcascade_eye_tree_eyeglasses.xml";
String face_model = "./train/trainner/trainner.yml";

CascadeClassifier face_cascade; // Declare the face classifier
CascadeClassifier eyes_cascade; // Declare the eyes classifier
Ptr<LBPHFaceRecognizer> model = Algorithm::load<LBPHFaceRecognizer>(face_model);//parameter Model

int main(int argc, char *argv[])
{
	/* Open the web camera */
	VideoCapture capture = VideoCapture(0);
	capture.set(CAP_PROP_FRAME_WIDTH, 1280); // 720p camera
	capture.set(CAP_PROP_FRAME_HEIGHT, 720);
	Mat frame, image;
	
	/** Load cascade classifiers **/
	if(!face_cascade.load(face_cascade_name))
	 	cout << "Error loading face cascade" << endl;
	if(!eyes_cascade.load(eyes_cascade_name))
	 	cout << "Error loading eye cascade" << endl;
	if(model->empty()){
		cout << "face model load fault!" << endl;
	}
	int64 app_start_time = getTickCount();

    int64 t = getTickCount();

	/** After the camera is opened **/
	if(capture.isOpened()){
	 	cout<<"Face Detection Started..."<<endl;
	 	for(;;){
     	 	int64 t = getTickCount();
	 	 	/* Get image from camera */
	 	 	capture>>frame;
	 	 	if(frame.empty()){
	 	  	 	cout << "ERROR capture frame" << endl;
	 	  	 	waitKey();
	 	  	 	break;	
	 	 	}
	 	 	/* Start the face detection function */
	 	 	DetectAndDraw(frame);
	 	 	cout << "time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;
	 	 	/** If you press ESC, q, or Q , the process will end **/
	 	 	char ch = (char)waitKey(1);
	 	 	if(ch==27 || ch=='q' || ch=='Q')
	 	 	 	break;
 	 	}
	}
	else
	 	cout << "ERROR open camera" << endl;
	return 0;
}

void DetectAndDraw(Mat frame){
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
    face_cascade.detectMultiScale(frame_gray, faces, 1.05, 5, CASCADE_SCALE_IMAGE, Size(30, 30)); // 1.1 - > 1.00

    /** Draw circles around the faces **/
    for (size_t i = 0; i < faces.size(); i++)
    {
     	String name;
        Point center;
        /* Draw rectangular on face */
        rectangle(frame, faces[i], Scalar(255, 0, 0), 3, 8, 0);
        int label;
	 	double confidence = 0;
        Mat faceROI = frame_gray(faces[i]);
	 	model->predict(faceROI, label, confidence); // predict myself model

	 	Point LB(faces[i].x, faces[i].y);
	 	if(confidence < 50){
	 	 	if(label == 1){
	 	 	 	name = "De Yu Hong";
	 	 	 	putText(frame, name, LB, FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0,0,255));
				cout << "De Yu Hong" << endl;
	 	 	}
	 	 	else
	 	 	 	cout << "XXXX" << endl;
	 	}
	 	else
		{
	 	 	name = "Unknown";
	 	 	putText(frame, name, LB, FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0,0,255));
	 	 	cout << "Unknown" << endl;
		}
	 	cout << confidence << endl;
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
	resize(frame,frame,Size(640,480),0,0,INTER_LINEAR);
    // Show Processed Image with detected faces
    imshow( "Face Detection", frame);
	waitKey(1);
}
