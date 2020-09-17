#include <royale.hpp>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <cstring>
#include <string>
#include <ctime>
#include <vector>
#include <algorithm>

struct CC_Component {
	CC_Component(int index_p, int x_start_p, int y_start_p, int width_p, int height_p)
		: index(index_p), x_start(x_start_p), y_start(y_start_p), width(width_p), height(height_p), buchstabe(0)
	{
		
	}

	int index;
	int x_start;
	int y_start;
	int width;
	int height;
	int buchstabe;
};

std::string type2str(int type)
{
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth)
	{
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

void drawHist(cv::Mat& img, const cv::Scalar& color, const std::string& win_name, const int& x_coord, const int& y_coord, const float divisor = 2.4f)
{
	//Histogrammgroesse
	const int Y_Histsize = 600;
	const int X_Histsize = 300;

	//Histogramm erstellen
	int histSize = 256;
	float range[] = { 0, 255 };
	const float* histRange = { range };
	cv::Mat hist;
	calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

	//Eigenschaften ausgeben
	double min, max;
	cv::minMaxLoc(hist, &min, &max);
	//std::cout << "Histogram-Minimum: " << min << "\n";
	//std::cout << "Histogram-Maximum: " << max << "\n\n";

	//Histogramm zeichnen
	cv::Mat hist_drawn(Y_Histsize, X_Histsize, CV_8UC3);
	hist_drawn = 0;

	for (int i = 0; i < hist.rows - 1; i++)
	{
		int x1 = i;
		int x2 = i + 1;
		int y1 = ((int)(hist.at<float>(i)) / -divisor) + Y_Histsize - 1;
		int y2 = ((int)(hist.at<float>(i + 1)) / -divisor) + Y_Histsize - 1;

		//Histogrammverlauf einzeichnen (Keypoints)
		cv::line(hist_drawn, cv::Point(x1, y1), cv::Point(x2, y2), color);

		//Histogramm ausmalen
		for (int j = y1; j < Y_Histsize - 1; j++)
		{
			cv::line(hist_drawn, cv::Point(x1, j), cv::Point(x1, j + 1), color);
		}
		for (int j = y2; j < Y_Histsize - 1; j++)
		{
			cv::line(hist_drawn, cv::Point(x1, j), cv::Point(x1, j + 1), color);
		}
	}

	//Histogramm ausgeben
	cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
	cv::imshow(win_name, hist_drawn);
	cv::moveWindow(win_name, x_coord, y_coord);
}

bool compareCC(CC_Component cc1, CC_Component cc2)
{
	if (cc1.x_start < cc2.x_start)
	{
		return true;
	} return false;
}

class MyListener : public royale::IDepthDataListener
{
private:
	//Livebild
	cv::Mat zImage, grayImage, cameraMatrix, distortionCoefficients;
	//Videoanalyse
	cv::Mat gray_gray, depth_color, depth_color_prev;
	//Variablen für CC-Analyse
	cv::Mat labels, stats, centroids;
	//Andere Variablen und Container
	std::mutex flagMutex;
	unsigned int frame_counter = 0;
	cv::Vec3b Farben[10];
	char Buchstaben[10] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'};
	std::vector<CC_Component> comps;
	std::string currentKey = "";

public:
	cv::VideoWriter video1, video2;

	void DrawCC(cv::Mat& grayFrame, cv::Mat& Components)
	{
		//Erstelle Windows
		/*cv::namedWindow("Diff", cv::WINDOW_NORMAL);
		cv::moveWindow("Diff", 1050, 0);
		cv::namedWindow("Binary", cv::WINDOW_NORMAL);
		cv::moveWindow("Binary", 1350, 0);
		cv::namedWindow("Colored", cv::WINDOW_NORMAL);
		cv::moveWindow("Colored", 1050, 300);*/

		//Erstelle Matritzen
		cv::Mat median, diff, binary; //binary = 8UC1 mit 224x170
		cv::Mat Graubild8U1C = grayFrame;
		
		//Konvertiere Grauwertbild in Farbbild
		//cv::cvtColor(grayFrame, grayFrame, cv::COLOR_GRAY2BGR);		

		//std::string type = type2str(grayFrame.type());
		//printf("Matrix: %s %dx%d \n", type.c_str(), grayFrame.cols, grayFrame.rows);
		cv::cvtColor(Graubild8U1C, Graubild8U1C, cv::COLOR_BGR2GRAY);

		//Erstelle Binär -und Differenzbild
		cv::medianBlur(Graubild8U1C, median, 21);		
		cv::subtract(median, Graubild8U1C, diff);		
		cv::threshold(diff, binary, 0, 255, cv::THRESH_OTSU);
		
		//Führe ConnectedComponents Analyse durch
		
		int nLabels = cv::connectedComponentsWithStats(255 - binary, labels, stats, centroids);
		Components = binary;

		for (int k = 1; k < stats.rows; k++)
		{
			if (stats.at<int>(cv::Point(0, k)) > 0 && stats.at<int>(cv::Point(1, k))) {
				
				int index = k;
				int x_start = stats.at<int>(cv::Point(0, k));
				int y_start = stats.at<int>(cv::Point(1, k));
				int width = stats.at<int>(cv::Point(2, k));
				int height = stats.at<int>(cv::Point(3, k));
			
				comps.emplace_back(CC_Component(index, x_start, y_start, width, height));
			}

			/*std::cout << std::endl << "Component: " << k << std::endl;
			std::cout << "X      = " << stats.at<int>(cv::Point(0, k)) << std::endl;
			std::cout << "Y      = " << stats.at<int>(cv::Point(1, k)) << std::endl;
			std::cout << "Width  = " << stats.at<int>(cv::Point(2, k)) << std::endl;
			std::cout << "Height = " << stats.at<int>(cv::Point(3, k)) << std::endl;*/
						
			//Felder liegen waagerecht
				//Width > 80 && Width < 90
				//Height > 10 && Height < 30
			if (stats.at<int>(cv::Point(2, k)) > 80 && stats.at<int>(cv::Point(2, k)) < 90 &&
				stats.at<int>(cv::Point(3, k)) > 10 && stats.at<int>(cv::Point(3, k)) < 30)
			{
				std::cout << std::endl << "Component gefunden! (waagerecht)";

				//Component in Bild einzeichnen
				int x_coord = stats.at<int>(cv::Point(0, k));
				int y_coord = stats.at<int>(cv::Point(1, k));
				int width = stats.at<int>(cv::Point(2, k));
				int height = stats.at<int>(cv::Point(3, k));						

				for (int i = x_coord + 1; i < x_coord + width - 1; i++)
				{
					for (int j = y_coord + 1; j < y_coord + height - 1; j++)
					{
						grayFrame.at<cv::Vec3b>(j, i) = Farben[k];
					}
				}
			}
			//Felder liegen senkrecht
			else if (stats.at<int>(cv::Point(3, k)) > 80 && stats.at<int>(cv::Point(3, k)) < 90 &&
					 stats.at<int>(cv::Point(2, k)) > 10 && stats.at<int>(cv::Point(2, k)) < 30)
			{
				std::cout << std::endl << "Component gefunden! (senkrecht)";

				//Component in Bild einzeichnen
				int x_coord = stats.at<int>(cv::Point(0, k));
				int y_coord = stats.at<int>(cv::Point(1, k));
				int width = stats.at<int>(cv::Point(2, k));
				int height = stats.at<int>(cv::Point(3, k));
				
				for (int i = x_coord + 1; i < x_coord + width - 1; i++)
				{
					for (int j = y_coord + 1; j < y_coord + height - 1; j++)
					{
						grayFrame.at<cv::Vec3b>(j, i) = Farben[k];
					}
				}
			}
		}

		/*cv::imshow("Colored", grayFrame);
		cv::imshow("Diff", diff);
		cv::imshow("Binary", binary);*/
	}

	void VideoAnalyse(const cv::Mat& Graubild8UC3, cv::Mat& Median8UC1, cv::Mat& Gauss8UC1)
	{
		//std::string type = type2str(GreyFrame1C.type());
		//printf("Matrix: %s %dx%d \n", type.c_str(), GreyFrame1C.cols, GreyFrame1C.rows);

		//Umwandlung in ein "wahres" 1-Channel Graubild
		cv::Mat Graubild8U1C = Graubild8UC3;
		cv::cvtColor(Graubild8U1C, Graubild8U1C, cv::COLOR_BGR2GRAY);

		//Median und Gau�filter anwenden
		cv::medianBlur(Graubild8U1C, Median8UC1, 7);
		cv::GaussianBlur(Graubild8U1C, Gauss8UC1, cv::Size(5, 5), 10.0f);

		//Histogramme erstellen und zeichnen
		drawHist(Graubild8U1C, cv::Scalar(0, 255, 0), "Original Grauwert-Histogramm", 0, 310);
		drawHist(Median8UC1, cv::Scalar(0, 0, 255), "Median Grauwert-Histogramm", 300, 310);
		drawHist(Gauss8UC1, cv::Scalar(255, 0, 0), "Gauss Grauwert-Histogramm", 600, 310);
	}

	void VideoAbspielen(std::string name1, std::string name2)
	{
		cv::VideoCapture cap1(name1);
		cv::VideoCapture cap2(name2);

		std::string window_name1 = name1;
		std::string window_name2 = name2;

		cv::namedWindow(window_name1, cv::WINDOW_NORMAL);
		cv::moveWindow(window_name1, 0, 0);
		cv::namedWindow(window_name2, cv::WINDOW_NORMAL);
		cv::moveWindow(window_name2, 0, 300);
		cv::namedWindow("PreviousDepthFrame", cv::WINDOW_NORMAL);
		cv::moveWindow("PreviousDepthFrame", 300, 300);
		cv::namedWindow("GrayColorFrame", cv::WINDOW_NORMAL);
		cv::moveWindow("GrayColorFrame", 300, 0);
		/*cv::namedWindow("CC-Frame", cv::WINDOW_NORMAL);
		cv::moveWindow("CC-Frame", 900, 0);*/

		if (cap1.isOpened() == false)
		{
			std::cout << "Video 1 konnte nicht geoeffnet werden" << std::endl;
			return;
		}

		if (cap2.isOpened() == false)
		{
			std::cout << "Video 2 konnte nicht geoeffnet werden" << std::endl;
			return;
		}

		cv::Mat ConnectedFrame, grayFrame;
		int counter = 0;

		for (;;)
		{
			counter++;			

			bool Success1 = cap1.read(gray_gray);
			depth_color_prev = depth_color.clone();
			bool Success2 = cap2.read(depth_color);

			if (Success1 == false || Success2 == false)
			{
				std::cout << "Fehler im Frame: Frame1: " << Success1 << ", Frame2: " << Success2 << std::endl;
				break;
			}
						
			if (Success2 == true && counter > 1)
			{
				TiefenBildAnalyse();				
				cv::imshow("PreviousDepthFrame", depth_color_prev);
			}
						
			if (counter == 15)
			{
				for (int i = 0; i < 10; i++) {
					cv::Vec3b color = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
					Farben[i] = color;
				}

				grayFrame = gray_gray;
				DrawCC(grayFrame, ConnectedFrame);
				std::sort(comps.begin(), comps.end(), compareCC);
				
				for(int i = 0; i < comps.size(); i++)
				{
					comps.at(i).buchstabe = i;
				}

				cv::imshow("GrayColorFrame", grayFrame);
				//cv::imshow("CC-Frame", ConnectedFrame);
			}

			cv::imshow(window_name1, gray_gray);
			cv::imshow(window_name2, depth_color);
			cv::waitKey(150);
		}
	}

	void TiefenBildAnalyse()
	{
		//Fenster erstellen
		cv::namedWindow("Compare_blau_CMP_GT", cv::WINDOW_NORMAL);
		cv::moveWindow("Compare_blau_CMP_GT", 600, 0);
		cv::namedWindow("Compare_rot_CMP_LT", cv::WINDOW_NORMAL);
		cv::moveWindow("Compare_rot_CMP_LT", 600, 300);
		cv::namedWindow("Compare_Color", cv::WINDOW_NORMAL);
		cv::moveWindow("Compare_Color", 900, 0);
		cv::namedWindow("Pressed_Key", cv::WINDOW_NORMAL);
		cv::moveWindow("Pressed_Key", 900, 300);

		//Tiefenfarbbild -> Tiefengraubild -> Tiefenbinärbild umwandeln
		cv::Mat depth_binaer = depth_color.clone();
		cv::cvtColor(depth_binaer, depth_binaer, cv::COLOR_BGR2GRAY);
		cv::threshold(depth_binaer, depth_binaer, 80, 255, cv::THRESH_BINARY);

		cv::Mat old_depth_binaer = depth_color_prev.clone();
		cv::cvtColor(old_depth_binaer, old_depth_binaer, cv::COLOR_BGR2GRAY);
		cv::threshold(old_depth_binaer, old_depth_binaer, 80, 255, cv::THRESH_BINARY);

		//Binärbilder kopieren
		cv::Mat rot = depth_binaer.clone();
		cv::Mat blau = depth_binaer.clone();

		//Vergleichen der Frames und reinschreiben in die Matritzen blau und rot
		compare(old_depth_binaer, depth_binaer, blau, cv::CMP_GT);
		compare(old_depth_binaer, depth_binaer, rot, cv::CMP_LT);

		//Opening auf beide Bilder anwenden
		cv::morphologyEx(rot, rot, cv::MORPH_ERODE, cv::getStructuringElement(1, cv::Size(7, 7)));
		cv::morphologyEx(rot, rot, cv::MORPH_DILATE, cv::getStructuringElement(1, cv::Size(3, 3)));
		cv::morphologyEx(blau, blau, cv::MORPH_ERODE, cv::getStructuringElement(1, cv::Size(7, 7)));
		cv::morphologyEx(blau, blau, cv::MORPH_DILATE, cv::getStructuringElement(1, cv::Size(3, 3)));

		//Original Grauwertbild kopieren
		cv::Mat color_compare = gray_gray.clone();

		//Text einsetzen
		cv::putText(color_compare, "A", cv::Point(color_compare.cols / 8, color_compare.rows / 5.5), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(30, 200, 30), 2);
		cv::putText(color_compare, "B", cv::Point(color_compare.cols / 4.5, color_compare.rows / 5.5), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(30, 200, 30), 2);
		cv::putText(color_compare, "C", cv::Point(color_compare.cols / 3.25, color_compare.rows / 5.5), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(30, 200, 30), 2);
		cv::putText(color_compare, "D", cv::Point(color_compare.cols / 2.5, color_compare.rows / 5.5), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(30, 200, 30), 2);
		cv::putText(color_compare, "E", cv::Point(color_compare.cols / 2, color_compare.rows / 5.5), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(30, 200, 30), 2);
		cv::putText(color_compare, "F", cv::Point(color_compare.cols / 1.65, color_compare.rows / 5.5), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(30, 200, 30), 2);
		cv::putText(color_compare, "G", cv::Point(color_compare.cols / 1.475, color_compare.rows / 5.5), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(30, 200, 30), 2);
		cv::putText(color_compare, "H", cv::Point(color_compare.cols / 1.30, color_compare.rows / 5.5), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(30, 200, 30), 2);
		
		//std::string type = type2str(color_compare.type());
		//printf("Matrix: %s %dx%d \n", type.c_str(), color_compare.cols, color_compare.rows);		

		//Matrix zum Zeichnen der gedrueckten Taste erstellen
		cv::Mat draw_key(color_compare.rows, color_compare.cols, CV_8UC3);
		draw_key = 0;

		for(unsigned int i = 35; i < 125; i++)
		{			
			for(unsigned j = 30; j < 200; j++)
			{			
				if (rot.at<uchar>(i, j) == 0 && blau.at<uchar>(i, j) == 255) {

					//Uebergang von schwarz zu weiss -> Finger geht runter
					color_compare.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);

					//Durchgehen der connected components
					for (CC_Component CC : comps)
					{
						int x_start = CC.x_start;
						int y_start = CC.y_start;
						int breite = CC.width;
						int hoehe = CC.height;

						//Liegt der Finger in einem Feld?
						if((i > y_start && i < y_start + hoehe) && (j > x_start && j < x_start + breite))
						{
							//Buchstaben des jeweiligen Feldes ausgeben
							std::cout << Buchstaben[CC.buchstabe] << "\n";
							currentKey = Buchstaben[CC.buchstabe];
						}
					}	
				}
				else if (rot.at<uchar>(i, j) == 255 && blau.at<uchar>(i, j) == 0) {

					//Uebergang von weiss zu schwarz -> Finger geht hoch
					color_compare.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
				}
			}			
		}

		//Buchstaben zeichnen
		cv::putText(draw_key, currentKey, cv::Point(color_compare.cols / 2.5, color_compare.rows / 1.5), cv::FONT_HERSHEY_DUPLEX, 3.0, CV_RGB(30, 200, 30), 4);
		
		//Ausgabe der Bilder
		cv::imshow("Compare_blau_CMP_GT", blau);
		cv::imshow("Compare_rot_CMP_LT", rot);	
		cv::imshow("Compare_Color", color_compare);
		cv::imshow("Pressed_Key", draw_key);
	}

	void onNewData(const royale::DepthData* data)
	{
		cv::namedWindow("Graubild", cv::WINDOW_NORMAL);
		cv::moveWindow("Graubild", 0, 0);
		cv::namedWindow("Tiefenbild", cv::WINDOW_NORMAL);
		cv::moveWindow("Tiefenbild", 224, 0);
		cv::namedWindow("Median", cv::WINDOW_NORMAL);
		cv::moveWindow("Median", 448, 0);
		cv::namedWindow("Gauss", cv::WINDOW_NORMAL);
		cv::moveWindow("Gauss", 750, 0);

		// this callback function will be called for every new depth frame
		double mintief, maxtief, mingrau, maxgrau;
		std::lock_guard<std::mutex> lock(flagMutex);
		zImage.create(cv::Size(data->width, data->height), CV_32FC1);
		grayImage.create(cv::Size(data->width, data->height), CV_32FC1);
		zImage = 0;
		grayImage = 0;
		int k = 0;

		for (int y = 0; y < zImage.rows; y++)
		{
			for (int x = 0; x < zImage.cols; x++)
			{
				auto curPoint = data->points.at(k);
				if (curPoint.depthConfidence > 0)
				{
					// if the point is valid
					zImage.at<float>(y, x) = curPoint.z;
					grayImage.at<float>(y, x) = curPoint.grayValue;
				}
				k++;
			}
		}

		float Skalar;
		cv::Mat comparison, zTrans, zFarb, MedianFrame, GaussFrame, grayFrame, Components;
		cv::Mat temp = zImage.clone();

		undistort(temp, zImage, cameraMatrix, distortionCoefficients);
		temp = grayImage.clone();

		undistort(temp, grayImage, cameraMatrix, distortionCoefficients);
		compare(zImage, 0, comparison, cv::CMP_GT); //Maske erstellen

		//Skaliere Tiefenbild
		minMaxLoc(zImage, &mintief, &maxtief, 0, 0, comparison); //Findet Minimum und Maximum Wert heraus
		Skalar = 255.0 / (maxtief - mintief);
		convertScaleAbs(zImage, zTrans, Skalar, -mintief); //Skaliert das Bild
		applyColorMap(zTrans, zFarb, cv::COLORMAP_RAINBOW); //Tiefenbild einfaerben			

		//Skaliere Graubild
		minMaxLoc(grayImage, &mingrau, &maxgrau, 0, 0, comparison); //Findet Minimum und Maximum Wert heraus
		Skalar = 255.0 / (maxgrau - mingrau);
		convertScaleAbs(grayImage, grayImage, Skalar, -mingrau); //Skaliert das Bild

		//Zeige Fenster an
		cv::imshow("Graubild", grayImage);
		cv::imshow("Tiefenbild", zFarb);

		//std::string type = type2str(grayImage.type());
		//printf("Matrix: %s %dx%d \n", type.c_str(), grayImage.cols, grayImage.rows);

		VideoAnalyse(grayImage, MedianFrame, GaussFrame);
		cv::imshow("Median", MedianFrame);
		cv::imshow("Gauss", GaussFrame);

		frame_counter++;
		if (frame_counter == 10) {
			for (int i = 0; i < 10; i++) {
				cv::Vec3b color = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
				Farben[i] = color;
			}

			grayFrame = grayImage;
			DrawCC(grayFrame, Components);

		}
		
		//Schreibe Videos in Dateien
		//video1.write(grayImage);
		//video2.write(zFarb);

		cv::waitKey(1);
	}

	void setLensParameters(const royale::LensParameters& lensParameters)
	{
		// Construct the camera matrix
		// (fx   0    cx)
		// (0    fy   cy)
		// (0    0    1 )
		cameraMatrix = (cv::Mat1d(3, 3) << lensParameters.focalLength.first, 0, lensParameters.principalPoint.first,
			0, lensParameters.focalLength.second, lensParameters.principalPoint.second,
			0, 0, 1);

		// Construct the distortion coefficients
		// k1 k2 p1 p2 k3
		distortionCoefficients = (cv::Mat1d(1, 5) << lensParameters.distortionRadial[0],
			lensParameters.distortionRadial[1],
			lensParameters.distortionTangential.first,
			lensParameters.distortionTangential.second,
			lensParameters.distortionRadial[2]);
	}
};

int main(int argc, char* argv[])
{
	srand(time(NULL));

	//Wenn ein Argument uebergeben wurde
	if (argc >= 2)
	{
		//Wenn das Argument eine 1 ist -> Auswertung starten
		if (!strcmp(argv[1], "1"))
		{
			std::cout << "Aufruf der Auswertung" << std::endl;
			std::cin.get();
		}

		//Wenn das Argument eine 2 ist -> Videoaufzeichnung starten
		else if (!strcmp(argv[1], "2"))
		{

			MyListener listener;

			// this represents the main camera device object
			std::unique_ptr<royale::ICameraDevice> cameraDevice;

			// the camera manager will query for a connected camera
			{
				royale::CameraManager manager;

				// try to open the first connected camera
				royale::Vector<royale::String> camlist(manager.getConnectedCameraList());
				std::cout << "Detected " << camlist.size() << " camera(s)." << std::endl;

				if (!camlist.empty())
				{
					cameraDevice = manager.createCamera(camlist[0]);
				}
				else
				{
					std::cerr << "No suitable camera device detected." << std::endl
						<< "Please make sure that a supported camera is plugged in, all drivers are "
						<< "installed, and you have proper USB permission" << std::endl;
					return 1;
				}

				camlist.clear();
			}

			// the camera device is now available and CameraManager can be deallocated here
			if (cameraDevice == nullptr)
			{
				// no cameraDevice available
				if (argc > 1)
				{
					std::cerr << "Could not open " << argv[1] << std::endl;
					return 1;
				}
				else
				{
					std::cerr << "Cannot create the camera device" << std::endl;
					return 1;
				}
			}

			// call the initialize method before working with the camera device
			auto status = cameraDevice->initialize();
			if (status != royale::CameraStatus::SUCCESS)
			{
				std::cerr << "Cannot initialize the camera device, error string : " << getErrorString(status) << std::endl;
				return 1;
			}

			// retrieve the lens parameters from Royale
			royale::LensParameters lensParameters;
			status = cameraDevice->getLensParameters(lensParameters);
			if (status != royale::CameraStatus::SUCCESS)
			{
				std::cerr << "Can't read out the lens parameters" << std::endl;
				return 1;
			}

			listener.setLensParameters(lensParameters);

			// register a data listener
			if (cameraDevice->registerDataListener(&listener) != royale::CameraStatus::SUCCESS)
			{
				std::cerr << "Error registering data listener" << std::endl;
				return 1;
			}

			uint16_t height, width, fps;
			std::string name1, name2;

			// Kamerattribute abfragen
			cameraDevice->getMaxSensorHeight(height);
			cameraDevice->getFrameRate(fps);
			cameraDevice->getMaxSensorWidth(width);
			cameraDevice->setExposureMode(royale::ExposureMode::AUTOMATIC);
			std::cout << "Height: " << height << " | Width: " << width << " | FPS: " << fps << "\n";

			//std::cout << "Bitte geben Sie den Namen fuer das erste Video an :" << std::endl;
			//std::cin >> name1;
			//std::cout << "Bitte geben Sie den Namen fuer das zweite Video an :" << std::endl;
			//std::cin >> name2;

			//std::string file_grey = "./" + name1 + "_gray.avi";
			//std::string file_depth = "./" + name2 + "_depth.avi";

			// Codec definieren
			//cv::VideoWriter video1(file_grey, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height), false);
			//cv::VideoWriter video2(file_depth, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height), true);

			// Codec-Parameter an on new data schicken
			//listener.video1 = video1;
			//listener.video2 = video2;

			// start capture mode
			if (cameraDevice->startCapture() != royale::CameraStatus::SUCCESS)
			{
				std::cerr << "Error starting the capturing" << std::endl;
				return 1;
			}

			bool running = true;
			std::string end;
			while (running == true)
			{
				std::cin >> end;
				if (end == "end")
					running = false;
			}

			// stop capture mode
			if (cameraDevice->stopCapture() != royale::CameraStatus::SUCCESS)
			{
				std::cerr << "Error stopping the capturing" << std::endl;
				return 1;
			}

			//listener.video1.release();
			//listener.video2.release();
		}

		//Wenn das Argument eine 3 ist -> Videoabspielmodus starten
		else if (!strcmp(argv[1], "3"))
		{
			std::string name1, name2;
			MyListener listener;
			std::cout << "Videoabspielmodus wird gestartet..." << std::endl << std::endl;
			/*std::cout << "Geben Sie den Namen des ersten Videos ein: " << std::endl;
			std::cin >> name1;
			std::cout << "Geben Sie den Namen des zweiten Videos ein: " << std::endl;
			std::cin >> name2;*/
			listener.VideoAbspielen(name1 = "example_gray.avi", name2 = "example_depth.avi");
		}

		//Aufruf ohne Argument -> Fehler
		else
		{
			std::cout << "Fehlerhafter Aufruf!" << std::endl;
		}
	}

	return 0;
}