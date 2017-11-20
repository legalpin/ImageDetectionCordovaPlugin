package com.cloudoki.imagedetectionplugin;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.*;
import android.graphics.Rect;
import android.hardware.Camera;
import android.os.Build;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Base64;
import android.util.Log;
import android.view.Gravity;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.FrameLayout;

import org.apache.cordova.CallbackContext;
import org.apache.cordova.CordovaInterface;
import org.apache.cordova.CordovaPlugin;
import org.apache.cordova.CordovaWebView;
import org.apache.cordova.PluginResult;
import org.json.JSONArray;
import org.json.JSONException;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

import static android.provider.ContactsContract.CommonDataKinds.StructuredName.SUFFIX;

public class ImageDetectionPlugin extends CordovaPlugin implements SurfaceHolder.Callback {

    private static final String  TAG = "OpenCV::Activity";
    private static final int     REQUEST_CAMERA_PERMISSIONS = 133;
    private static final int CAMERA_ID_ANY   = -1;
    private static final int CAMERA_ID_BACK  = 99;
    private static final int CAMERA_ID_FRONT = 98;

    @SuppressWarnings("deprecation")
    private Camera               camera;
    private Activity             activity;
    private SurfaceHolder        surfaceHolder;
    private Mat                  mYuv;
    private Mat                  mYuvOrig;
    private Mat                  desc2;
    private FeatureDetector      orbDetector;
    private DescriptorExtractor  orbDescriptor;
    private MatOfKeyPoint        kp2;
    private MatOfDMatch          matches;
    private CallbackContext      cb;
    private long                 last_time;
    private boolean processFrames = true, thread_over = true, debug = false,
            called_success_detection = false, called_failed_detection = true,
            previewing = false, save_files = false;
    private List<Integer> detection = new ArrayList<>();

    private List<Mat> triggers = new ArrayList<>();
    private List<MatOfKeyPoint> triggers_kps = new ArrayList<>();
    private List<Mat> triggers_descs = new ArrayList<>();
    private int trigger_size = -1, detected_index = -1;

    private long timeout = 500;
    private int cameraId = -1;
    private int mCameraIndex = CAMERA_ID_ANY;

    private BaseLoaderCallback mLoaderCallback;
    private FrameLayout cameraFrameLayout;

    private  int count = 0;
    private String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    private String coords;
    private int screenWidth = 1, screenHeight = 1;

    // Mios
    private CascadeClassifier faceCascade;
    private CascadeClassifier eyeCascade;
    private Mat               mGrayFrame;

    final private int DEFAULT_CAMERA = CAMERA_ID_FRONT;
    @SuppressWarnings("deprecation")
    private static class JavaCameraSizeAccessor implements CameraBridgeViewBase.ListItemAccessor {

        @Override
        public int getWidth(Object obj) {
            Camera.Size size = (Camera.Size) obj;
            return size.width;
        }

        @Override
        public int getHeight(Object obj) {
            Camera.Size size = (Camera.Size) obj;
            return size.height;
        }
    }

    @Override
    public void initialize(CordovaInterface cordova, CordovaWebView webView) {
        activity = cordova.getActivity();

        super.initialize(cordova, webView);

        mLoaderCallback = new BaseLoaderCallback(activity) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:
                    {
                        Log.i(TAG, "OpenCV loaded successfully");
                    } break;
                    default:
                    {
                        super.onManagerConnected(status);
                    } break;
                }
            }
        };

        activity.getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        activity.getWindow().setFormat(PixelFormat.TRANSLUCENT);
        activity.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);

        SurfaceView surfaceView = new SurfaceView(activity.getApplicationContext());
        surfaceView.setBackgroundColor(Color.WHITE);

        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
                Gravity.CENTER);

        cameraFrameLayout = new FrameLayout(activity.getApplicationContext());

        activity.getWindow().addContentView(cameraFrameLayout, params);

        surfaceHolder = surfaceView.getHolder();
        surfaceHolder.addCallback(this);

        cameraFrameLayout.addView(surfaceView);
        cameraFrameLayout.setVisibility(View.VISIBLE);

        sendViewToBack(cameraFrameLayout);

        setCameraIndex(DEFAULT_CAMERA);

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        AssetManager am = activity.getAssets();

        try {
            faceCascade = new CascadeClassifier(stream2file(am.open("data/haarcascades/haarcascade_frontalface_default.xml")).getPath());
            eyeCascade = new CascadeClassifier(stream2file(am.open("data/haarcascades/haarcascade_eye.xml")).getPath());
            Log.d(TAG, " *******************************************: Cassifiers loaded OK");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static File stream2file (InputStream in) throws IOException {
        final String PREFIX = "idplugin";

        final File tempFile = File.createTempFile(PREFIX, null);
        tempFile.deleteOnExit();
        OutputStream out = null;
        try {
            out = new FileOutputStream(tempFile);
            copyStream(in, out);
        } catch (Exception e) {
            try {
                if (out != null)
                    out.close();
            } catch (Exception e2) {
            }
            out=null;
        }
        return tempFile;
    }

    public static void copyStream(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[4096];
        int read;
        while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }
    }

    private void listFiles(String path) {
        Log.d(TAG, " ******************************************* Listing files in path: "+path);
        File folder = new File(path);
        File[] listOfFiles = folder.listFiles();

        if(listOfFiles ==  null ) {
            Log.d(TAG, " ******************************************* NO FILES");
            return;
        }
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                Log.d(TAG, " ******************************************* File " + listOfFiles[i].getName());
            } else if (listOfFiles[i].isDirectory()) {
                Log.d(TAG, " ******************************************* Directory " + listOfFiles[i].getName());
            }
        }

    }
    @Override
    public boolean execute(String action, JSONArray data,
                           CallbackContext callbackContext) throws JSONException {

        if (action.equals("openCamera")) {
            openCamera();
            return true;
        }

        if (action.equals("closeCamera")) {
            closeCamera();
            return true;
        }

        if (action.equals("greet")) {
            Log.i(TAG, "greet called");
            String name = data.getString(0);
            if(name != null && !name.isEmpty()) {
                String message = "Hello, " + name;
                callbackContext.success(message);
            } else {
                callbackContext.error("");
            }
            return true;
        }
        if (action.equals("isDetecting")) {
            Log.i(TAG, "isDetecting called");
            cb = callbackContext;
            return true;
        }
        if(action.equals("setPatterns")) {
            Log.i(TAG, "setPatterns called");
            final JSONArray inputData = data;
            final CallbackContext cbContext = callbackContext;
            cordova.getThreadPool().execute(new Runnable() {
                public void run() {
                    // clear before adding triggers
                    triggers.clear();
                    triggers_kps.clear();
                    triggers_descs.clear();

                    String message = "Pattens to be set - " + inputData.length();
                    message += "\nBefore set pattern " + triggers.size();
                    setBase64Pattern(inputData);
                    message += "\nAfter set pattern " + triggers.size();
                    if(inputData.length() == triggers.size()) {
                        trigger_size = triggers.size();
                        message += "\nPatterns set - " + triggers.size();
                        cbContext.success(message);
                    } else {
                        message += "\nOne or more patterns failed to be set.";
                        cbContext.error(message);
                    }
                }
            });
            return true;
        }
        if(action.equals("startProcessing")) {
            Log.i(TAG, "startProcessing called");
            String message;
            boolean argVal;
            try {
                argVal = data.getBoolean(0);
                screenHeight = data.getInt(1);
                screenWidth = data.getInt(2);
            } catch (JSONException je) {
                argVal = true;
                Log.e(TAG, je.getMessage());
            }
            if(argVal) {
                processFrames = true;
                message = "Frame processing set to 'true', screen size is h - " + screenHeight + ", w - " + screenWidth;
                callbackContext.success(message);
            } else {
                processFrames = false;
                message = "Frame processing set to 'false', screen size is h - " + screenHeight + ", w - " + screenWidth;
                callbackContext.error(message);
            }
            return true;
        }
        if(action.equals("setDetectionTimeout")) {
            Log.i(TAG, "setDetectionTimeout called");
            String message;
            long argVal;
            try {
                argVal = data.getLong(0);
            } catch (JSONException je) {
                argVal = -1;
                Log.e(TAG, je.getMessage());
            }
            if(argVal >= 0) {
                timeout = argVal;
                message = "Processing timeout set to " + timeout;
                callbackContext.success(message);
            } else {
                message = "No value or timeout value negative.";
                callbackContext.error(message);
            }
            return true;
        }
        return false;
    }

    private void closeCamera() {
        if(camera == null)
            return;

        File sdCard = Environment.getExternalStorageDirectory();

        if(mYuv!=null)
            Imgcodecs.imwrite(sdCard.getAbsolutePath() + "/img_yuv.jpeg", mYuv);

        if(mGrayFrame!=null)
            Imgcodecs.imwrite(sdCard.getAbsolutePath() + "/img_gray.jpeg", mGrayFrame);

        camera.setPreviewCallback(null);
        camera.stopPreview();
        camera.release();
        previewing = false;
        camera = null;
        setCameraBackground(true);
    }

    private void setCameraBackground(final boolean white ) {
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                cameraFrameLayout.removeAllViews();

                SurfaceView surfaceView = new SurfaceView(activity.getApplicationContext());
                if(white)
                    surfaceView.setBackgroundColor(Color.WHITE);

                FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        Gravity.CENTER);

                cameraFrameLayout = new FrameLayout(activity.getApplicationContext());
                cameraFrameLayout.setVisibility(View.VISIBLE);
                cameraFrameLayout.addView(surfaceView);

                activity.getWindow().addContentView(cameraFrameLayout, params);

                surfaceHolder = surfaceView.getHolder();
                sendViewToBack(cameraFrameLayout);

                setCameraIndex(DEFAULT_CAMERA);
            }
        });
    }

    @Override
    public void onStart()
    {
        super.onStart();

        Log.i(TAG, "onStart(): Activity starting");

        if(!checkCameraPermission()) {
            ActivityCompat.requestPermissions(activity,
                    new String[]{Manifest.permission.CAMERA},
                    REQUEST_CAMERA_PERMISSIONS);
        }

        if(save_files) {
            int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);

            if (permission != PackageManager.PERMISSION_GRANTED) {
                // We don't have permission so prompt the user
                int REQUEST_EXTERNAL_STORAGE = 1;
                ActivityCompat.requestPermissions(
                        activity,
                        PERMISSIONS_STORAGE,
                        REQUEST_EXTERNAL_STORAGE
                );
            }
        }

        thread_over = true;
        debug = false;
        called_success_detection = false;
        called_failed_detection = true;

        last_time = 0;
    }

    public static void sendViewToBack(final View child) {
        final ViewGroup parent = (ViewGroup)child.getParent();
        if (null != parent) {
            parent.removeView(child);
            parent.addView(child, 0);
        }
    }

    private boolean checkCameraPermission() {
        return ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onPause(boolean multitasking) {
        super.onPause(multitasking);
    }

    @Override
    public void onResume(boolean multitasking) {
        super.onResume(multitasking);

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");

            //OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, activity, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

/*
        if (camera == null) {
            openCamera();
            camera.stopPreview();
        }
*/
    }

    @Override
    public void onStop() {
        super.onStop();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }

    private void setWhiteBackground(SurfaceHolder holder) {

        Canvas canvas = holder.lockCanvas();
        if (canvas != null) {
            canvas.drawRGB(255, 255, 255);
            holder.unlockCanvasAndPost(canvas);
        }

    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        matches = new MatOfDMatch();
        orbDetector = FeatureDetector.create(FeatureDetector.ORB);
        orbDescriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        kp2 = new MatOfKeyPoint();
        desc2 = new Mat();
        //setWhiteBackground(holder);
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        if(previewing){
            camera.stopPreview();
            previewing = false;
        }

        if (camera != null){
            boolean result = initializeCamera(height, width);
            if( !result ) {
                AlertDialog.Builder builder = new AlertDialog.Builder(activity);
                builder.setTitle("An error occurred")
                        .setMessage("An error occurred while trying to open the camera.")
                        .setCancelable(false)
                        .setPositiveButton("Ok", new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                activity.finish();
                            }
                        });
                AlertDialog alert = builder.create();
                alert.show();
            }
            previewing = true;
        }

        //setWhiteBackground(holder);

    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        if (camera != null) {
            camera.setPreviewCallback(null);
            camera.stopPreview();
            camera.release();
            camera = null;
            previewing = false;
        }
    }

    private void setCameraIndex(int index) {
        mCameraIndex = index;
    }

    @SuppressWarnings("deprecation")
    private void openCamera() {
        if(camera != null)
            closeCamera();

        setCameraBackground(false);

        camera = null;

        if (mCameraIndex == CAMERA_ID_ANY) {
            Log.d(TAG, "Trying to open camera with old open()");
            try {
                camera = Camera.open();
            } catch (Exception e) {
                Log.e(TAG, "Camera is not available (in use or does not exist): " + e.getLocalizedMessage());
            }

            if (camera == null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.GINGERBREAD) {
                boolean connected = false;
                for (int camIdx = 0; camIdx < Camera.getNumberOfCameras(); ++camIdx) {
                    Log.d(TAG, "Trying to open camera with new open(" + camIdx + ")");
                    try {
                        camera = Camera.open(camIdx);
                        connected = true;
                    } catch (RuntimeException e) {
                        Log.e(TAG, "Camera #" + camIdx + "failed to open: " + e.getLocalizedMessage());
                    }
                    if (connected) break;
                }
            }
        } else {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.GINGERBREAD) {
                int localCameraIndex = mCameraIndex;
                if (mCameraIndex == CAMERA_ID_BACK) {
                    Log.i(TAG, "Trying to open back camera");
                    Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
                    for (int camIdx = 0; camIdx < Camera.getNumberOfCameras(); ++camIdx) {
                        Camera.getCameraInfo(camIdx, cameraInfo);
                        if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_BACK) {
                            localCameraIndex = camIdx;
                            break;
                        }
                    }
                } else if (mCameraIndex == CAMERA_ID_FRONT) {
                    Log.i(TAG, "Trying to open front camera");
                    Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
                    for (int camIdx = 0; camIdx < Camera.getNumberOfCameras(); ++camIdx) {
                        Camera.getCameraInfo(camIdx, cameraInfo);
                        if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                            localCameraIndex = camIdx;
                            break;
                        }
                    }
                }
                if (localCameraIndex == CAMERA_ID_BACK) {
                    Log.e(TAG, "Back camera not found!");
                } else if (localCameraIndex == CAMERA_ID_FRONT) {
                    Log.e(TAG, "Front camera not found!");
                } else {
                    Log.d(TAG, "Trying to open camera with new open(" + localCameraIndex + ")");
                    try {
                        camera = Camera.open(localCameraIndex);
                    } catch (RuntimeException e) {
                        Log.e(TAG, "Camera #" + localCameraIndex + "failed to open: " + e.getLocalizedMessage());
                    }
                }
                cameraId = localCameraIndex;
            }
        }

        Rect rect = surfaceHolder.getSurfaceFrame();
        initializeCamera(rect.height(), rect.width());
        if(camera != null) {
            try {
                camera.setPreviewDisplay(surfaceHolder);
            } catch (IOException e) {
                e.printStackTrace();
            }
            camera.startPreview();
        }
    }

    @SuppressWarnings("deprecation")
    private boolean initializeCamera(int height, int width) {
        boolean result = true;
        synchronized (this) {
            if (camera == null)
                return false;

            /* Now set camera parameters */
            try {
                Camera.Parameters params = camera.getParameters();
                Log.d(TAG, "getSupportedPreviewSizes()");
                List<Camera.Size> sizes = params.getSupportedPreviewSizes();

                if (sizes != null) {
                    /* Select the size that fits surface considering maximum size allowed */
                    Size frameSize = calculateCameraFrameSize(sizes, new JavaCameraSizeAccessor(), width, height);

                    params.setPreviewFormat(ImageFormat.NV21);
                    Log.d(TAG, "Set preview size to " + frameSize.width + "x" + frameSize.height);
                    params.setPreviewSize((int)frameSize.width, (int)frameSize.height);

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.ICE_CREAM_SANDWICH && !android.os.Build.MODEL.equals("GT-I9100"))
                        params.setRecordingHint(true);

                    List<String> FocusModes = params.getSupportedFocusModes();
                    if (FocusModes != null && FocusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO))
                    {
                        Log.d(TAG, "Set focus mode continuous video " + Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO );
                        params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
                    }
                    else if(FocusModes != null && FocusModes.contains(Camera.Parameters.FOCUS_MODE_AUTO)) {
                        Log.d(TAG, "Set focus mode auto " + Camera.Parameters.FOCUS_MODE_AUTO );
                        params.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
                    }

                    if(activity != null) {
                        Camera.CameraInfo info = new Camera.CameraInfo();
                        Camera.getCameraInfo(cameraId, info);
                        int cameraRotationOffset = info.orientation;

                        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
                        int degrees = 0;
                        switch (rotation) {
                            case Surface.ROTATION_0:
                                degrees = 0;
                                break; // Natural orientation
                            case Surface.ROTATION_90:
                                degrees = 90;
                                break; // Landscape left
                            case Surface.ROTATION_180:
                                degrees = 180;
                                break;// Upside down
                            case Surface.ROTATION_270:
                                degrees = 270;
                                break;// Landscape right
                        }
                        int displayRotation;
                        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                            displayRotation = (cameraRotationOffset + degrees) % 360;
                            displayRotation = (360 - displayRotation) % 360; // compensate the mirror
                        } else { // back-facing
                            displayRotation = (cameraRotationOffset - degrees + 360) % 360;
                        }

                        Log.v(TAG, "rotation cam / phone = displayRotation: " + cameraRotationOffset + " / " + degrees + " = "
                                + displayRotation);

                        camera.setDisplayOrientation(displayRotation);

                        int rotate;
                        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                            rotate = (360 + cameraRotationOffset + degrees) % 360;
                        } else {
                            rotate = (360 + cameraRotationOffset - degrees) % 360;
                        }

                        Log.v(TAG, "screenshot rotation: " + cameraRotationOffset + " / " + degrees + " = " + rotate);

                        params.setRotation(rotate);

                        params.setPreviewFrameRate(24);// set camera preview

                        camera.setParameters(params);
                        //camera.setPreviewDisplay(surfaceHolder);
                        camera.setPreviewCallback(previewCallback);
                    }

                    /* Finally we are ready to start the preview */
                    Log.d(TAG, "startPreview");
                    //camera.startPreview();
                }
                else
                    result = false;
            } catch (Exception e) {
                result = false;
                e.printStackTrace();
            }
        }

        return result;
    }

    private Size calculateCameraFrameSize(List<?> supportedSizes, CameraBridgeViewBase.ListItemAccessor accessor, int surfaceHeight, int surfaceWidth) {
        int calcWidth = 0;
        int calcHeight = 0;

        for (Object size : supportedSizes) {
            int width = accessor.getWidth(size);
            int height = accessor.getHeight(size);

            if (width <= surfaceWidth && height <= surfaceHeight) {
                if (width >= calcWidth && height >= calcHeight) {
                    calcWidth = width;
                    calcHeight = height;
                }
            }
        }

        return new Size(calcWidth, calcHeight);
    }

    @SuppressWarnings("deprecation")
    private final Camera.PreviewCallback previewCallback = new Camera.PreviewCallback() {
        @Override
        public void onPreviewFrame(byte[] data, Camera camera) {
            //Log.d(TAG, "ON Preview frame: "+System.currentTimeMillis());

            long current_time = System.currentTimeMillis();
            long time_passed = current_time - last_time;

            if(processFrames && time_passed > timeout) {
                Log.d(TAG, "Processing frame: "+System.currentTimeMillis());
                if (thread_over) {
                    thread_over = false;
                    //Log.d(TAG, "Processing frame inner: "+System.currentTimeMillis());
                    Camera.Parameters params = camera.getParameters();

                    int height = params.getPreviewSize().height;
                    int width  = params.getPreviewSize().width;

                    mYuvOrig = new Mat(height, width, CvType.CV_8UC1);
                    mYuv = new Mat(width, height, CvType.CV_8UC1);

                    mYuvOrig.put(0, 0, data);
                    Core.transpose(mYuvOrig, mYuv);
                    Core.flip(mYuv, mYuv, -1);

                    final MatOfRect faces = new MatOfRect();
                    final int absoluteFaceSize = Math.round(mYuv.height() * 0.2f);

                    faceCascade.detectMultiScale(mYuv, faces, 1.3, 2, Objdetect.CASCADE_SCALE_IMAGE, new Size(absoluteFaceSize, absoluteFaceSize), new Size());

                    org.opencv.core.Rect[] facesArray = faces.toArray();
                    int facesLength = facesArray.length;
                    if (facesLength > 0)
                        Log.d(TAG, "Faces array size: " + facesLength);

                    thread_over = true;
                }
                //update time and reset timeout
                last_time = current_time;
            }

        }
    };

    private void setBase64Pattern(JSONArray dataArray) {
        detection.clear();
        for (int i = 0; i < dataArray.length(); i++) {
            try {
                detection.add(0);
                String image_base64 = dataArray.getString(i);
                if(image_base64 != null && !image_base64.isEmpty()) {
                    Mat image_pattern = new Mat();
                    MatOfKeyPoint kp1 = new MatOfKeyPoint();
                    Mat desc1 = new Mat();

                    int limit = 400;
                    if(image_base64.contains("data:"))
                        image_base64 = image_base64.split(",")[1];
                    byte[] decodedString = Base64.decode(image_base64, Base64.DEFAULT);
                    Bitmap bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
                    Bitmap scaled = bitmap;
                    if (bitmap.getWidth() > limit) {
                        double scale = bitmap.getWidth() / limit;
                        scaled = Bitmap.createScaledBitmap(bitmap, (int) (bitmap.getWidth()/scale), (int) (bitmap.getHeight()/scale), true);
                        if (bitmap.getHeight() > limit) {
                            scale = bitmap.getHeight() / limit;
                            scaled = Bitmap.createScaledBitmap(bitmap, (int) (bitmap.getWidth()/scale), (int) (bitmap.getHeight()/scale), true);
                        }
                    }
                    Utils.bitmapToMat(scaled, image_pattern);
                    Imgproc.cvtColor(image_pattern, image_pattern, Imgproc.COLOR_BGR2GRAY);
                    //Imgproc.equalizeHist(image_pattern, image_pattern);

                    if(save_files) {
                        Utils.matToBitmap(image_pattern, scaled);
                        String extStorageDirectory = Environment.getExternalStorageDirectory().toString();
                        int num = (int) (Math.random() * 10001);
                        Imgcodecs.imwrite(extStorageDirectory + "/pic" + num + ".png", image_pattern);
                        Log.i("### FILE ###", "File saved to " + extStorageDirectory + "/pic" + num + ".png");
                    }

                    orbDetector.detect(image_pattern, kp1);
                    orbDescriptor.compute(image_pattern, kp1, desc1);

                    triggers.add(image_pattern);
                    triggers_kps.add(kp1);
                    triggers_descs.add(desc1);
                }
            } catch (JSONException e) {
                // do nothing
            }
        }
    }

    private void updateState(boolean state, int _index) {
        final int index = _index;

        int detection_limit = 6;

        if(state) {
            try {
                int result = detection.get(_index) + 1;
                if(result < detection_limit) {
                    detection.set(_index, result);
                }
            } catch (IndexOutOfBoundsException ibe){
//                 detection.add(_index, 1);
            }
        } else {
            for (int i = 0; i < triggers.size(); i++) {
                try {
                    int result = detection.get(i) - 1;
                    if(result < 0) {
                        detection.set(_index, 0);
                    } else {
                        detection.set(_index, result);
                    }
                } catch (IndexOutOfBoundsException ibe){
//                    detection.add(i, 0);
                }
            }
        }

        if (getState(_index) && called_failed_detection && !called_success_detection) {
            cordova.getThreadPool().execute(new Runnable() {
                public void run() {
                    PluginResult result = new PluginResult(PluginResult.Status.OK, "{\"message\":\"pattern detected\", \"index\":" + index + ", " + coords + "}");
                    result.setKeepCallback(true);
                    cb.sendPluginResult(result);
                }
            });
            called_success_detection = true;
            called_failed_detection = false;
            detected_index = _index;
        }

        boolean valid_index = detected_index == _index;

        if (!getState(_index) && !called_failed_detection && called_success_detection && valid_index) {
            cordova.getThreadPool().execute(new Runnable() {
                public void run() {
                    PluginResult result = new PluginResult(PluginResult.Status.ERROR, "{\"message\":\"pattern not detected\"}");
                    result.setKeepCallback(true);
                    cb.sendPluginResult(result);
                }
            });
            called_success_detection = false;
            called_failed_detection = true;
        }
    }

    private boolean getState(int index) {
        int total;
        int detection_thresh = 3;

        total = detection.get(index);

        if(debug) {
            Log.i("## GET STATE RESULT ##", " state -> " + total);
        }

        return total >= detection_thresh;
    }
}
