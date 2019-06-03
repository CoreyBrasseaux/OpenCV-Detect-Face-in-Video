import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.BackgroundSubtractor;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

class FaceInVideo {
    public void run(String[] args) {

        String cmlFile = "xml/lbpcascade_frontalface_improved.xml";
        CascadeClassifier cc = new CascadeClassifier((cmlFile));

        String input = args.length > 0 ? args[0] : "data/vtest2.mp4";
        boolean useMOG2 = args.length > 1 ? args[1] == "MOG2" : true;
        BackgroundSubtractor backSub;
        if (useMOG2) {
            backSub = Video.createBackgroundSubtractorMOG2();
            System.out.println("using MOG2");
        } else {
            backSub = Video.createBackgroundSubtractorKNN();
            System.out.println("using KNN");
        }
        VideoCapture capture = new VideoCapture(input);
        if (!capture.isOpened()) {
            System.err.println("Unable to open: " + input);
            System.exit(0);
        }
        Mat frame = new Mat();
        MatOfRect faceDetection = new MatOfRect();
        //, fgMask = new Mat();
        while (true) {
            capture.read(frame);
            if (frame.empty()) {
                break;
            }
            // update the background model
            //backSub.apply(frame, fgMask);
            // get the frame number and write it on the current frame
            Imgproc.rectangle(frame, new Point(10, 2), new Point(100, 20), new Scalar(255, 255, 255), -1);
            String frameNumberString = String.format("%d", (int)capture.get(Videoio.CAP_PROP_POS_FRAMES));

            cc.detectMultiScale(frame, faceDetection);
            for(Rect rect: faceDetection.toArray()){
                Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 0, 255), 3);

            }


            //font face was originally core.FONT_HERSHEY_SIMPLEX but was causing error
            Imgproc.putText(frame, frameNumberString, new Point(15, 15), 0, 0.5,
                    new Scalar(0, 0, 0));
            // show the current frame and the fg masks
            HighGui.imshow("Frame", frame);
            //HighGui.imshow("FG Mask", fgMask);
            // get the input from the keyboard
            int keyboard = HighGui.waitKey(30);
            if (keyboard == 'q' || keyboard == 27) {
                break;
            }
        }
        HighGui.waitKey();
        System.exit(0);
    }
}

public class FaceInVideoDemo {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new FaceInVideo().run(args);
    }
}
