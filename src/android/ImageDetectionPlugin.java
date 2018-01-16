package com.legalpin.imagedetectionplugin;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.PixelFormat;
import android.hardware.Camera;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
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
import android.widget.Toast;

import org.apache.cordova.CallbackContext;
import org.apache.cordova.CordovaInterface;
import org.apache.cordova.CordovaPlugin;
import org.apache.cordova.CordovaWebView;
import org.apache.cordova.PluginResult;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Predicate;

import com.googlecode.tesseract.android.TessBaseAPI;
import com.legalpin.facelib.NativeMethods;

import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.THRESH_OTSU;

public class ImageDetectionPlugin extends CordovaPlugin implements SurfaceHolder.Callback {

    private static final String  TAG = "OpenCV::Activity";
    private static final int REQUEST_CAMERA_PERMISSIONS = 133;
    private static final int CAMERA_ID_ANY   = -1;
    private static final int CAMERA_ID_BACK  = 99;
    private static final int CAMERA_ID_FRONT = 98;

    @SuppressWarnings("deprecation")
    private Camera               camera;
    private Activity             activity;
    private SurfaceHolder        surfaceHolder;
    private Mat                  mYuv;
    private Mat                  mYuvOrig;
    private CallbackContext      cb;
    private long                 last_time;
    private boolean processFrames = true, thread_over = true,
            previewing = false, save_files = false;

    private long timeout = 250;
    private int cameraId = -1;
    private int mCameraIndex = CAMERA_ID_ANY;

    private BaseLoaderCallback mLoaderCallback;
    private FrameLayout cameraFrameLayout;

    private  int count = 0;
    private String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    // Mios
    private CascadeClassifier faceCascade;
    private NativeMethods.TrainFacesTask mTrainFacesTask;
    private ArrayList<Mat> imagesFaces;
    private ArrayList<Mat> imagesFacesTmp;
    private ArrayList<String> facesList;
    private boolean useEigenfaces = false;
    private boolean training = false;
    //private boolean detecting = false;
    private boolean detectingFace = false;
    private boolean detectingDocumentFront  = false;
    private boolean detectingDocumentBack = false;

    private String currentTrainingFace = null;

    private int numAttempts = 0;

    private static final double FACE_THRESHOLD     = 0.033;
    private static final double DISTANCE_THRESHOLD = 0.007;
    private static final int NUM_USER_FACES        = 10;
    private static final int MAX_ATTEMPTS          = 10;

    private static final int PROCESS_MODE_DETECT_FACE = 1;
    private static final int PROCESS_MODE_TRAIN_FACE  = 2;
    private static final int PROCESS_MODE_RECOG_FACE  = 3;
    private static final int PROCESS_MODE_FRONT_ID    = 4;
    private static final int PROCESS_MODE_BACK_ID     = 5;


    private TessBaseAPI tessBaseApi;

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
        System.loadLibrary("face-lib");
        System.loadLibrary("tess");
        System.loadLibrary("lept");
        System.loadLibrary("jpgt");
        System.loadLibrary("pngt");

        AssetManager am = activity.getAssets();

        try {
            faceCascade = new CascadeClassifier(stream2file(am.open("data/haarcascades/haarcascade_frontalface_default.xml")).getPath());
            Log.d(TAG, "Cassifiers loaded OK");
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

    @Override
    public boolean execute(String action, JSONArray data,
                           CallbackContext callbackContext) throws JSONException {
        Log.i(TAG, " ********************** execute: action="+action);

        if (action.equals("openCamera")) {
            cb = callbackContext;
            this.processFrames = false;
            this.training = false;

            int type = -1;
            try {
                type = data.getInt(0);
            } catch (JSONException je) {
            }

            if (type == 1 || type == -1)
                setCameraIndex(CAMERA_ID_FRONT);
            else if(type == 2)
                setCameraIndex(CAMERA_ID_BACK);
            else
                setCameraIndex(CAMERA_ID_ANY);

            openCamera();
            cb.success();

            return true;
        }

        if (action.equals("closeCamera")) {
            cb = callbackContext;
            closeCamera();
            return true;
        }

        if (action.equals("startTraining")) {
            cb = callbackContext;
            String faceName = null;
            try {
                faceName = data.getString(0);
            } catch (JSONException je) {
                Log.e(TAG, je.getMessage());
                je.printStackTrace();
            }

            if(faceName==null || "".equals(faceName)) {
                cb.error("FAIL2");
                return true;
            }

            startTraining(faceName);

            return true;
        }

        if (action.equals("isTraining")) {
            Log.i(TAG, "isTraining called");
            PluginResult result = new PluginResult(PluginResult.Status.OK, this.training);
            result.setKeepCallback(false);
            callbackContext.sendPluginResult(result);

            return true;
        }

        if(action.equals("startDetecting")) {
            cb = callbackContext;

            Log.d(TAG, " ******************************* startDetecting called");

            int type = -1;
            try {
                type = data.getInt(0);
            } catch (JSONException je) {
            }

            Log.d(TAG, " ******************************* startDetecting: "+type);
            if (type == 1 || type == -1)
                startDetectingFace();
            else if (type == 2)
                startDetectingDocumentFront();
            else if (type == 3)
                startDetectingDocumentBack();
/*
            else {
                PluginResult result = new PluginResult(PluginResult.Status.ERROR);
                result.setKeepCallback(false);
                callbackContext.sendPluginResult(result);
            }
*/
            return true;
        }

        if(action.equals("stopDetecting")) {
            Log.i(TAG, "stopDetecting called");
            this.processFrames = false;
            this.detectingFace = false;
            this.detectingDocumentFront = false;
            this.detectingDocumentBack = false;

            PluginResult result = new PluginResult(PluginResult.Status.OK);
            result.setKeepCallback(false);
            callbackContext.sendPluginResult(result);
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

    private void startTraining(String faceName) {
        Log.d(TAG, "startTraining(): called");
        if(imagesFaces==null)
            imagesFaces = new ArrayList<Mat>();

        if(facesList==null)
            facesList = new ArrayList<>();

        imagesFacesTmp = new ArrayList();

        PluginResult pluginResult = new  PluginResult(PluginResult.Status.NO_RESULT);
        pluginResult.setKeepCallback(true); // Keep callback
        cb.sendPluginResult(pluginResult);

        if(faceExists(faceName)) {
            cb.error("FAIL3");
            return;
        }

        currentTrainingFace = faceName;

        this.detectingFace = false;
        this.training = true;
        this.processFrames = true;
    }

    private boolean faceExists(String faceName) {
        for(String face: facesList)
            if(face.equalsIgnoreCase(faceName))
                return true;

        return false;
    }

    private void startDetectingFace() {
        Log.d(TAG, "startDetectingFace(): called");

        if(facesList.size()==0) {
            cb.error("NO FACES");
            return;
        }

        PluginResult pluginResult = new  PluginResult(PluginResult.Status.NO_RESULT);
        pluginResult.setKeepCallback(true); // Keep callback
        cb.sendPluginResult(pluginResult);


        this.numAttempts = 0;
        this.processFrames = true;
        this.training = false;
        this.detectingFace = true;
    }

    private void startDetectingDocumentFront() {
        Log.d(TAG, " ********************************* startDetectingDocumentFront(): called");

        PluginResult pluginResult = new  PluginResult(PluginResult.Status.NO_RESULT);
        pluginResult.setKeepCallback(true); // Keep callback
        cb.sendPluginResult(pluginResult);
        this.detectingDocumentFront = true;
        this.processFrames = true;
    }

    private void startDetectingDocumentBack() {
        Log.d(TAG, "startDetectingDocumentBack(): called");

        PluginResult pluginResult = new  PluginResult(PluginResult.Status.NO_RESULT);
        pluginResult.setKeepCallback(true); // Keep callback
        cb.sendPluginResult(pluginResult);
        this.detectingDocumentBack = true;
        this.processFrames = true;
    }

    private void closeCamera() {
        if(camera == null)
            return;

/*
        File sdCard = Environment.getExternalStorageDirectory();
        if(mYuv!=null)
            Imgcodecs.imwrite(sdCard.getAbsolutePath() + "/img_yuv.jpeg", mYuv);
*/

        camera.setPreviewCallback(null);
        camera.stopPreview();
        camera.release();
        camera = null;

        processFrames = false;
        detectingFace = false;
        training = false;

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

        Log.d(TAG, "onStart(): Activity starting");

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

    private int getCameraIndex() {
        return mCameraIndex;
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

        android.graphics.Rect rect = surfaceHolder.getSurfaceFrame();
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
                //Log.d(TAG, "Processing frame: "+System.currentTimeMillis());
                if (thread_over) {
                    thread_over = false;
                    Camera.Parameters params = camera.getParameters();

                    int height = params.getPreviewSize().height;
                    int width  = params.getPreviewSize().width;

                    mYuvOrig = new Mat(height, width, CvType.CV_8UC1);
                    mYuv = new Mat(width, height, CvType.CV_8UC1);

                    mYuvOrig.put(0, 0, data);
                    
                    saveImg("test1.jpeg", mYuvOrig);
                    if(training || detectingFace) {
                        Core.transpose(mYuvOrig, mYuv);
                        Core.flip(mYuv, mYuv, -1);

                        int minDimension = height > width ? width : height;

                        final MatOfRect faces = new MatOfRect();
                        final int absoluteFaceSize = Math.round(minDimension * 0.5f);
                        final int minValidFaceSize = Math.round(minDimension * 0.75f);

                        //faceCascade.detectMultiScale(mYuv, faces, 1.3, 2, Objdetect.CASCADE_SCALE_IMAGE, new Size(absoluteFaceSize, absoluteFaceSize), new Size());
                        faceCascade.detectMultiScale(mYuv, faces, 1.3, 5, Objdetect.CASCADE_SCALE_IMAGE, new Size(absoluteFaceSize, absoluteFaceSize), new Size());

                        org.opencv.core.Rect[] facesArray = faces.toArray();

                        final int facesLength = facesArray.length;
                        if (facesLength == 1 && facesArray[0].height >= minValidFaceSize) {
                            org.opencv.core.Rect face = facesArray[0];
                            Log.d(TAG,"FACE CASCADE CLASSIFIER OK!: IH:"+mYuv.height()+"AFS:" +absoluteFaceSize+" H:"+face.height+" W:"+face.width);

                            if(training) {
                                saveFace(mYuv);
                                //Log.d(TAG, "Faces array size: " + facesLength);
                            }

                            if(detectingFace) {
                                detectFace(mYuv);
                                //Log.d(TAG, "Faces array size: " + facesLength);
                            }
                        }
                    } else {
                        mYuv = mYuvOrig;
                        if(detectingDocumentFront)
                            detectDocumentFront(mYuv);

                        if(detectingDocumentBack)
                            detectDocumentBack(mYuv);
                    }

                    thread_over = true;
                }
                //update time and reset timeout
                last_time = current_time;
            }

        }
    };

    private void saveFace(Mat image_pattern) {
        try {
            int numFace = imagesFacesTmp.size();
            Log.d(TAG, "Saving face #" + imagesFacesTmp.size());
            //String extStorageDirectory = Environment.getExternalStorageDirectory().toString();
            //File lpinDir = new File(extStorageDirectory + "/lpin");
            //lpinDir.mkdir();
            //Imgcodecs.imwrite(extStorageDirectory + "/lpin/pic_" + numFace + ".png", image_pattern);

            Mat image = image_pattern.reshape(0, (int) image_pattern.total()); // Create column vector


            if (imagesFacesTmp.size() >= NUM_USER_FACES) {
                training = false;
                imagesFaces.addAll(imagesFacesTmp);
                facesList.add(currentTrainingFace);
                trainFaces();
                return;
            }

            JSONObject json = new JSONObject();
            json.put("numFaces", imagesFacesTmp.size()+1);
            json.put("totalFaces", NUM_USER_FACES);

            PluginResult result = new PluginResult(PluginResult.Status.OK, json);
            result.setKeepCallback(true);
            cb.sendPluginResult(result);
            imagesFacesTmp.add(image);

        } catch (Exception e) {
            //PluginResult result = new PluginResult(PluginResult.Status.ERROR);
            //result.setKeepCallback(false);
            //cb.sendPluginResult(result);
            cb.error(e.getClass().getName()+": "+e.getMessage());
        }
    }

    /**
     * Train faces using stored images.
     * @return  Returns false if the task is already running.
     */
    private boolean trainFaces() {
        try {
            JSONObject json = new JSONObject();
            json.put("processingImages","Y");

            PluginResult result = new PluginResult(PluginResult.Status.OK, json);
            result.setKeepCallback(true);
            cb.sendPluginResult(result);
        } catch (JSONException e) {
            e.printStackTrace();
        }

        Log.d(TAG, "trainFaces(): CALLED");
        if (imagesFaces.isEmpty())
            return true; // The array might be empty if the method is changed in the OnClickListener

        if (mTrainFacesTask != null && mTrainFacesTask.getStatus() != AsyncTask.Status.FINISHED) {
            Log.i(TAG, "mTrainFacesTask is still running");
            return false;
        }

        Mat imagesMatrix = new Mat((int) imagesFaces.get(0).total(), imagesFaces.size(), imagesFaces.get(0).type());
        for (int i = 0; i < imagesFaces.size(); i++)
            imagesFaces.get(i).copyTo(imagesMatrix.col(i)); // Create matrix where each image is represented as a column vector

        Log.d(TAG, "Images height: " + imagesMatrix.height() + " Width: " + imagesMatrix.width() + " total: " + imagesMatrix.total());

        // Train the face recognition algorithms in an asynchronous task, so we do not skip any frames
        if (useEigenfaces) {
            Log.i(TAG, "Training Eigenfaces");

            mTrainFacesTask = new NativeMethods.TrainFacesTask(imagesMatrix, trainFacesTaskCallback);
        } else {
            Log.i(TAG, "Training Fisherfaces");

            int[] classes = new int[imagesFaces.size()];
            for (int i = 0; i < imagesFaces.size(); i++) {
                int classNumber = ( (i * 2) / NUM_USER_FACES )+1;
                Log.d(TAG, "Image: "+i+" class: "+classNumber);
                classes[i] = classNumber; // Insert corresponding number
            }

            Mat vectorClasses = new Mat(classes.length, 1, CvType.CV_32S); // CV_32S == int
            vectorClasses.put(0, 0, classes); // Copy int array into a vector

            mTrainFacesTask = new NativeMethods.TrainFacesTask(imagesMatrix, vectorClasses, trainFacesTaskCallback);
        }
        mTrainFacesTask.execute();

        return true;
    }

    private NativeMethods.TrainFacesTask.Callback trainFacesTaskCallback = new NativeMethods.TrainFacesTask.Callback() {
        @Override
        public void onTrainFacesComplete(boolean result) {
            if (result) {
                Log.d(TAG, "Training complete");

                PluginResult pluginResult = new PluginResult(PluginResult.Status.OK);
                pluginResult.setKeepCallback(false);
                cb.sendPluginResult(pluginResult);
            } else {
                Log.d(TAG, "Training failed");
                PluginResult pluginResult = new PluginResult(PluginResult.Status.ERROR);
                pluginResult.setKeepCallback(false);
                cb.sendPluginResult(pluginResult);
            }
        }
    };


    public void detectFace(Mat mGray) {
        Log.d(TAG, "detectFace() called");
        NativeMethods.MeasureDistTask mMeasureDistTask = null;

        if (mMeasureDistTask != null && mMeasureDistTask.getStatus() != AsyncTask.Status.FINISHED) {
            Log.d(TAG, "mMeasureDistTask is still running");
            return;
        }
        if (mTrainFacesTask != null && mTrainFacesTask.getStatus() != AsyncTask.Status.FINISHED) {
            Log.d(TAG, "mTrainFacesTask is still running");
            return;
        }

        Log.d(TAG, "Gray height: " + mGray.height() + " Width: " + mGray.width() + " total: " + mGray.total());
        if (mGray.total() == 0)
            return;

        //Size imageSize = new Size(200, 200.0f / ((float) mGray.width() / (float) mGray.height())); // Scale image in order to decrease computation time
        //Imgproc.resize(mGray, mGray, imageSize);

        Log.d(TAG, "Small gray height: " + mGray.height() + " Width: " + mGray.width() + " total: " + mGray.total());
        //SaveImage(mGray);

        Mat image = mGray.reshape(0, (int) mGray.total()); // Create column vector
        Log.d(TAG, "Vector height: " + image.height() + " Width: " + image.width() + " total: " + image.total());

        // Calculate normalized Euclidean distance
        mMeasureDistTask = new NativeMethods.MeasureDistTask(useEigenfaces, measureDistTaskCallback);
        mMeasureDistTask.execute(image);
        //mMeasureDistTask.execute(mGray);
    }

    public void detectDocumentFront(Mat mGray) {
        //Log.d(TAG, "detectDocumentFront() called");

        double ratio = mGray.rows() / (double) mGray.cols();
        //Log.d(TAG, "detectDocumentFront(): ratio: "+ratio);
        int RESIZE = 500;

        //int height = mGray.cols();
        //int width  = mGray.rows();

        int newHeight = RESIZE;
        int newWidth  = (int) (RESIZE * ratio);

        //Log.d(TAG, "detectDocumentFront(): resize: ("+width+","+height+") -> ("+newWidth+","+newHeight+")");

        //Mat mSmall = new Mat(new Size(newHeight, newWidth), mGray.type());
        Mat mSmall = new Mat();
        Size sz = new Size(newHeight, newWidth);
        Imgproc.resize( mGray, mSmall, sz );

        double sharpness = getSharpness(mGray);
        Log.d(TAG, "Image sharpness: "+sharpness);

        if(sharpness < 110) {
            Log.d(TAG, "Blurry image not valid");
            return;
        }

        Mat mCanny = getEdgesFromImage(mSmall);
        MatOfPoint2f contoursSmall = findContours(mCanny);
        if(contoursSmall == null)
            return;

        MatOfPoint2f contoursOK = correctPerspective(contoursSmall, mSmall.size());
        if(contoursOK.get(0,0)[0]!=contoursSmall.get(0,0)[0] ||
                contoursOK.get(0,0)[1]!=contoursSmall.get(0,0)[1])
            Log.d(TAG, " ********************************** PERSPECTIVE CORRECTION contoursSmall: "+getContourToString(contoursSmall)+" contoursOK: "+getContourToString(contoursOK));

        MatOfPoint2f bigContours = recalculateContours(mSmall.size(), mGray.size(), contoursOK);

        //Log.d(TAG, "detectDocumentFront(): rectangle detected");

        Mat mWarpedSmall = applyTransformationRotation(mSmall, contoursOK);
        if(mWarpedSmall==null) {
            Log.d(TAG, "No contours found!");
            return;
        }

        Mat mWarpedBig = applyTransformationRotation(mGray, bigContours);

        //double angle1 = computeSkew(mWarped, 32);
/*
        List<Double> lAngle = new ArrayList();

        for(int i = 96; i <= 192; i += 8)
            lAngle.add(computeSkew(mWarped, i));

        double bestAngle = selectBestAngle(lAngle);
*/
//        for(int j = 0; j<lAngle.size();j++)
//            Log.d(TAG, "Skew angle[" + j + "]: " + lAngle.get(j));

//        Log.d(TAG, "Best angle: " + bestAngle);
        //Mat mDeskewed = deskew(mWarped, bestAngle);

        saveImg("x1.jpeg", mGray);
        saveImg("x2.jpeg", mSmall);
        saveImg("x3.jpeg", mCanny);
        saveImg("x4.jpeg", mWarpedSmall);
        saveImg("x5.jpeg", mWarpedBig);

        Rect rectSmall = detectMRZ(mWarpedSmall);
        if(rectSmall!=null) {
            Rect rectBig = recalculateRect(mWarpedSmall.size(), mWarpedBig.size(), rectSmall);

            if (rectSmall != null) {
                String text = getImageToString(mWarpedBig, rectBig);

                Log.d(TAG, "TEXTO EXTRAIDO:\n" + text);
            }
        }
    }

    private double getSharpness(Mat matGray) {
        Mat destination = new Mat();
        //Mat matGray=new Mat();

        //Imgproc.cvtColor(image, matGray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.Laplacian(matGray, destination, 3);
        MatOfDouble median = new MatOfDouble();
        MatOfDouble std= new MatOfDouble();
        Core.meanStdDev(destination, median , std);

        return Math.pow(std.get(0,0)[0],2);
    }

    private Rect recalculateRect(Size org, Size dst, Rect orgRect) {

        double ratioX = dst.height / org.height;
        double ratioY = dst.width / org.width;


        Rect dstRect = new Rect();
        dstRect.x = (int) (orgRect.x * ratioX);
        dstRect.y = (int) (orgRect.y * ratioY);

        dstRect.width = (int) (orgRect.width * ratioX);
        dstRect.height = (int) (orgRect.height * ratioY);


        Log.d(TAG, "Size: ORG: W: "+org.width+" H: "+org.height+" DST: W: "+dst.width+" H: "+dst.height);

        Log.d(TAG, "recalculateRect: ("+orgRect.x+","+orgRect.y+"),("+(orgRect.x+orgRect.width)+","+(orgRect.y+orgRect.height)+") ->  ("+dstRect.x+","+dstRect.y+"),("+(dstRect.x+dstRect.width)+","+(dstRect.y+dstRect.height)+")");

        return dstRect;
    }

    private MatOfPoint2f recalculateContours(Size sOrg, Size sDst, MatOfPoint2f contours) {

        final double srcHeight = sOrg.height;
        final double srcWidth = sOrg.width;

        final double dstHeight = sDst.height;
        final double dstWidth = sDst.width;

        final double ratioX = dstWidth / srcWidth;
        final double ratioY = dstHeight / srcHeight;

        final List<Point> lDstContours = new ArrayList();

        for(Point p : contours.toArray())
            lDstContours.add(new Point(ratioX * p.x, ratioY * p.y));

        final MatOfPoint2f result = new MatOfPoint2f();
        result.fromList(lDstContours);

        return result;
    }

    private String getImageToString(Mat image, Rect rect) {
        Mat image_roi = new Mat(image, rect);

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1; // 1 - means max size. 4 - means maxsize/4 size. Don't use value <4, because you need more memory in the heap to store your data.
        saveImg("temp.jpeg", image_roi);

        File sdCard = Environment.getExternalStorageDirectory();

        Bitmap bitmap = BitmapFactory.decodeFile(sdCard.getPath()+"/temp.jpeg", options);

        return extractText(bitmap);
    }

    private String extractText(Bitmap bitmap) {
        try {
            tessBaseApi = new TessBaseAPI();
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            if (tessBaseApi == null) {
                Log.e(TAG, "TessBaseAPI is null. TessFactory not returning tess object.");
            }
        }
/*
File outputDir = context.getCacheDir(); // context being the Activity pointer
File outputFile = File.createTempFile("prefix", "extension", outputDir);
 */

        tessBaseApi.init(Environment.getExternalStorageDirectory().getAbsolutePath(), "eng");
//       //EXTRA SETTINGS
//        //For example if we only want to detect numbers
//        tessBaseApi.setVariable(TessBaseAPI.VAR_CHAR_WHITELIST, "1234567890");
//
//        //blackList Example
//        tessBaseApi.setVariable(TessBaseAPI.VAR_CHAR_BLACKLIST, "!@#$%^&*()_+=-qwertyuiop[]}{POIU" +
//                "YTRWQasdASDfghFGHjklJKLl;L:'\"\\|~`xcvXCVbnmBNM,./<>?");

        Log.d(TAG, "Training file loaded");
        tessBaseApi.setImage(bitmap);
        String extractedText = "empty result";
        try {
            extractedText = tessBaseApi.getUTF8Text();
        } catch (Exception e) {
            Log.e(TAG, "Error in recognizing text.");
        }
        tessBaseApi.end();
        return extractedText;
    }


    private Rect detectMRZ(Mat image) {
        saveImg("test2.jpeg",image);
        //Log.d(TAG, "X1");
        Mat rectKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(13,5));
        Mat sqKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(21,21));
        //Log.d(TAG, "X2");
        //Mat gray = new Mat(image.rows(), image.cols(), image.type());
        //Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        //Log.d(TAG, "X3");
        Mat gray = image.clone();
        Imgproc.GaussianBlur(gray, gray, new Size(3,3),0);
        Mat blackhat = new Mat();
        Imgproc.morphologyEx(gray, blackhat, Imgproc.MORPH_BLACKHAT, rectKernel);
        //Log.d(TAG, "X4");
        Mat gradX = new Mat(1,1,CvType.CV_8UC1);
        Imgproc.Sobel(blackhat, gradX, CvType.CV_8UC1, 1,0); // -1
        //Log.d(TAG, "X5");

        Core.absdiff(gradX, new Scalar(0), gradX);
        //absoluteMat(gradX, gradX);
        Core.MinMaxLocResult mmlr = Core.minMaxLoc(gradX);

                /*
        double minVal = minMat(gradX);
        double maxVal = maxMat(gradX);

        */
        //Log.d(TAG, "X6");


        //gradMinMax(gradX, gradX, minVal, maxVal);
        gradMinMax(gradX, gradX, mmlr.minVal, mmlr.maxVal);

        Mat thres = gradX.clone();
        //Log.d(TAG, "X7");
        Imgproc.morphologyEx(gradX, gradX, Imgproc.MORPH_CLOSE, rectKernel);
        Imgproc.threshold(gradX, thres, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        Imgproc.morphologyEx(thres, thres, Imgproc.MORPH_CLOSE, sqKernel);
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.erode(thres, thres, element);

        List<MatOfPoint> cnts = new ArrayList();
        //Log.d(TAG, "X8");
        Imgproc.findContours(thres.clone(),  cnts, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        //Log.d(TAG, "X9");
        cnts.sort(new Comparator<MatOfPoint>() {
            public int compare(MatOfPoint mop1, MatOfPoint mop2) {
                double aMop1 = Imgproc.contourArea(mop1);
                double aMop2 = Imgproc.contourArea(mop2);
                if(aMop1 < aMop2)
                    return 1;

                if(aMop1 > aMop2)
                    return -1;

                return 0;
            }
        });
        //Log.d(TAG, "X10");
        Rect result = null;
        for(MatOfPoint c : cnts) {
            Rect rect = Imgproc.boundingRect(c);
            Point p = rect.tl();
            double x = p.x;
            double y = p.y;
            double w = rect.width;
            double h = rect.height;
            double ar = w / h;
            double crWidth = w / gray.cols();

            Log.d(TAG, "ar: "+ar+" crWidth: "+crWidth);
            if(ar > 5 && crWidth > 0.75) {
                int pX = (int) ((x + w) * 0.03);
                int pY = (int) ((y + h) * 0.03);

                x = x - pX;
                y = y - pY;
                w = w + (pX * 2);
                h = h + (pY * 2);

                x = (x < 0) ? 0 : x;
                y = (y < 0) ? 0 : y;

                result = new Rect((int) x, (int) y,  (int) w,  (int) h);
                Mat imageRect = image.clone();
                Imgproc.rectangle(imageRect, new Point(x,y),new Point(x+w,y+h), new Scalar(100));
                saveImg("test3.jpeg", imageRect);
                break;
            }

        }
        //Log.d(TAG, "X11");
        if(result!=null)
            Log.d(TAG, "Rectangle: "+result.toString());
        else
            Log.d(TAG, "Rectangle is null!");
        //Log.d(TAG, "X12");
        return result;

    }

    private void gradMinMax(Mat mSrc, Mat mDst, double minVal, double maxVal) {
        int cols = mSrc.cols();
        int rows = mSrc.rows();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double[] src = mSrc.get(i, j);
                double[] dst = new double[src.length];
                for (int k = 0; k < src.length; k++)
                    dst[k] = (int) (255 * ((src[k] - minVal) / (maxVal - minVal)));

                mDst.put(i, j, dst);
            }
        }
    }

    private double minMat(Mat mat) {
        double minMat = Double.MAX_VALUE;
        int cols = mat.cols();
        int rows = mat.rows();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double[] src = mat.get(i, j);
                for (int k = 0; k < src.length; k++) {
                    if (src[k] < minMat)
                        minMat = src[i];
                }
            }
        }

        return minMat;
    }

    private double maxMat(Mat mat) {
        double maxMat = Double.MIN_VALUE;
        int cols = mat.cols();
        int rows = mat.rows();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double[] src = mat.get(i, j);
                for (int k = 0; k < src.length; k++) {
                    if (src[k] > maxMat)
                        maxMat = src[k];
                }
            }
        }

        return maxMat;
    }

    private void absoluteMat(Mat mSrc, Mat mDst) {
        int cols = mSrc.cols();
        int rows = mSrc.rows();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double[] src = mSrc.get(i, j);
                double[] dst = new double[src.length];
                for (int k = 0; k < src.length; k++)
                    dst[k] = Math.abs(src[k]);

                mDst.put(i, j, dst);
            }
        }
    }

    private double selectBestAngle(List<Double> lAngle) {
        double selAngle = Double.MAX_VALUE;

        for (double curAngle : lAngle) {
            double absAngle = Math.abs(curAngle);
            if (absAngle < 5 && absAngle > 0 && absAngle < Math.abs(selAngle))
                selAngle = curAngle;
        }

        if (selAngle == Double.MAX_VALUE)
            selAngle = 0.0;

        return selAngle;
    }

    private Mat deskew(Mat src, double angle) {
        Point center = new Point(src.width()/2, src.height()/2);
        Mat rotImage = Imgproc.getRotationMatrix2D(center, angle, 1.0);
        //1.0 means 100 % scale
        Size size = new Size(src.width(), src.height());
        Mat dst = new Mat(src.rows(),src.cols(),src.type());

        Imgproc.warpAffine(src.clone(), dst, rotImage, size, Imgproc.INTER_LINEAR | Imgproc.CV_WARP_FILL_OUTLIERS, Core.BORDER_REPLICATE, new Scalar(0));
        return dst;
    }

    private double computeSkew( Mat src, double thres ) {
        //Log.d(TAG, "B1");
        Mat img = new Mat(src.rows(), src.cols(), src.type());
        //Core.multiply(img, new Scalar(2.0), img);
        //Binarize it
        //Use adaptive threshold if necessary
        //Imgproc.adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 40);
        //Log.d(TAG, "B2");
        Imgproc.threshold( src.clone(), img, thres, 255, THRESH_BINARY );

        //saveImg("x41.jpeg", img);
        //Invert the colors (because objects are represented as white pixels, and the background is represented by black pixels)
        //Log.d(TAG, "B3");
        Core.bitwise_not( img, img );
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));

        //Log.d(TAG, "B4");
        //We can now perform our erosion, we must declare our rectangle-shaped structuring element and call the erode function
        Imgproc.erode(img, img, element);
        //saveImg("x42.jpeg", img);

        //Log.d(TAG, "B5");
        //Find all white pixels
        Mat wLocMat = Mat.zeros(img.size(),img.type());
        Core.findNonZero(img, wLocMat);

        //Log.d(TAG, "B6");
        //Create an empty Mat and pass it to the function
        MatOfPoint matOfPoint = new MatOfPoint( wLocMat );

        //Log.d(TAG, "B7");
        //Translate MatOfPoint to MatOfPoint2f in order to user at a next step
        MatOfPoint2f mat2f = new MatOfPoint2f();
        matOfPoint.convertTo(mat2f, CvType.CV_32FC2);

        if(mat2f.toArray().length==0)
            return 0.0;

        //Get rotated rect of white pixels
        RotatedRect rotatedRect = Imgproc.minAreaRect( mat2f );

        Point[] vertices = new Point[4];
        rotatedRect.points(vertices);
        //Log.d(TAG, "B9");
/*
        List<MatOfPoint> boxContours = new ArrayList<>();
        boxContours.add(new MatOfPoint(vertices));
        Imgproc.drawContours( img, boxContours, 0, new Scalar(128, 128, 128), -1);
        //saveImg("x43.jpeg", img);

*/
        //Log.d(TAG, "B10");
        double resultAngle = rotatedRect.angle;
        if (resultAngle < -45.0)
            resultAngle = -(90.0 + resultAngle);

        return resultAngle;
    }

    private void saveImg(String name, Mat img) {
        Log.d(TAG, "Saving img: "+name);
        File sdCard = Environment.getExternalStorageDirectory();
        Imgcodecs.imwrite(sdCard.getAbsolutePath() + "/"+name, img);
    }

    private Mat getEdgesFromImage(Mat mGray) {
        Mat mGaussian = new Mat(mGray.rows(),mGray.cols(),mGray.type());
        //Mat mMul = mGaussian.clone();
        //Mat mAdd = mGaussian.clone();
        Mat mCanny = mGaussian.clone();

        //Core.multiply(mGray, new Scalar(2.0), mMul);
        //Core.add(mMul, new Scalar(50.0), mAdd);

        //Imgproc.GaussianBlur(mAdd, mGaussian, new Size(5,5), 0);
        //Imgproc.GaussianBlur(mMul, mGaussian, new Size(5,5), 0);
        Imgproc.GaussianBlur(mGray, mGaussian, new Size(5,5), 0);
        Imgproc.Canny(mGaussian, mCanny, 75, 200);

        return mCanny;
    }

    @SuppressLint("NewApi")
    private MatOfPoint2f findContours(Mat mCanny) {
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(mCanny.clone(), contours, new Mat(),  Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        contours.removeIf(new Predicate<MatOfPoint>() {
            @Override
            public boolean test(MatOfPoint mop) {
                return Imgproc.contourArea(mop) < 1000;
            }
        });

        contours.sort(new Comparator<MatOfPoint>() {
            public int compare(MatOfPoint mop1, MatOfPoint mop2) {
                double aMop1 = Imgproc.contourArea(mop1);
                double aMop2 = Imgproc.contourArea(mop2);
                if(aMop1 < aMop2)
                    return 1;

                if(aMop1 > aMop2)
                    return -1;

                return 0;
            }
        });

/*
        int i=0;
        for(MatOfPoint contour : contours) {
            Log.d(TAG, " * Contour area["+(i++)+"]: "+Imgproc.contourArea(contour));
        }
*/

        if(contours.size()>5)
            contours = contours.subList(0, 5);

        for(MatOfPoint c : contours) {
            //Minimun size allowed for consideration
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            MatOfPoint2f contour2f = new MatOfPoint2f(c.toArray());


            //Processing on mMOP2f1 which is in type MatOfPoint2f
            double approxDistance = Imgproc.arcLength(contour2f,true)*0.02;
            Imgproc.approxPolyDP(contour2f,approxCurve,approxDistance,true);

            //convert to MatofPoint
            if(approxCurve.toArray().length == 4)
                return approxCurve;
        }

        return null;
    }


    private MatOfPoint2f correctPerspective(MatOfPoint2f src, Size size) {

        final List<Point> lPointSrc = src.toList();
        if(lPointSrc.size()!=4)
            return null;

        final double halfX = size.width / 2;
        final double halfY = size.height / 2;

        Point point1 = lPointSrc.get(0);

        List<Point> lPointDst = null;

        if(point1.x < halfX && point1.y < halfY) {
            lPointDst = new ArrayList(lPointSrc.size());
            Point pointFinal = lPointSrc.get(3);
            lPointDst.add(pointFinal);
            for(int i=0;i<3;i++)
                lPointDst.add(lPointSrc.get(i));
        } else {
            lPointDst = lPointSrc;
        }

        MatOfPoint2f result = new MatOfPoint2f();
        result.fromList(lPointDst);

        return result;
    };


    private Mat applyTransformationRotation(Mat srcImage, MatOfPoint2f src) {
        Mat destImage = new Mat(srcImage.rows(), srcImage.cols(), srcImage.type());

        int x = destImage.width()-1;
        int y = destImage.height()-1;

        int minArea = (int) (x * y * 0.40);
        int area = (int) Imgproc.contourArea(src);
        Log.d(TAG, " * applyTransformationRotation(): area = " + Imgproc.contourArea(src)+" minArea = "+minArea);
        if(area < minArea)
            return null;

        Log.d(TAG, " * applyTransformationRotation(): contour = "+getContourToString(src));

        //Log.d(TAG, " * area = " + Imgproc.contourArea(src)+" minArea = "+minArea);

        Mat dst = new MatOfPoint2f(new Point(x, 0), new Point(0,0), new Point(0, y), new Point(x, y));

        Mat trans = Imgproc.getPerspectiveTransform(src, dst);

        Imgproc.warpPerspective(srcImage, destImage, trans, destImage.size());

        //double ratioWarped = destImage.cols()/destImage.rows();

        return destImage;
    }

    private String getContourToString(MatOfPoint2f src) {
        StringBuilder sb = new StringBuilder();
        for(Point p : src.toArray()) {
            sb.append(p.toString());
        }

        return sb.toString();
    };

    public void detectDocumentBack(Mat mGray) {
        Log.d(TAG, "detectDocumentFront() called");
        NativeMethods.MeasureDistTask mMeasureDistTask = null;
    }

    private NativeMethods.MeasureDistTask.Callback measureDistTaskCallback = new NativeMethods.MeasureDistTask.Callback() {
        @Override
        public void onMeasureDistComplete(Bundle bundle) {
            if (bundle == null) {
                Log.d(TAG, "Failed to measure distance: "+ Toast.LENGTH_LONG);
                return;
            }

            double faceThreshold = FACE_THRESHOLD;
            double distanceThreshold = DISTANCE_THRESHOLD;
            float minDist = bundle.getFloat(NativeMethods.MeasureDistTask.MIN_DIST_FLOAT);

            numAttempts++;

            Log.d(TAG, "Measure distance callback ["+numAttempts+"]: "+ bundle.toString());

            if (minDist != -1) {
                int minIndex = bundle.getInt(NativeMethods.MeasureDistTask.MIN_DIST_INDEX_INT);
                float faceDist = bundle.getFloat(NativeMethods.MeasureDistTask.DIST_FACE_FLOAT);
                if (faceDist < faceThreshold && minDist < distanceThreshold) {// 1. Near face space and near a face class
                    //Log.d(TAG, "Face detected: " + imagesLabels.get(minIndex) + ". Distance: " + minDistString);
                    Log.d(TAG, "Face detected! minIndex: " + minIndex);
                    int numFace = (minIndex / NUM_USER_FACES);
                    finishDetection(facesList.get(numFace), true);
                } else if (faceDist < faceThreshold) { // 2. Near face space but not near a known face class
                    //Log.d(TAG, "Unknown face. Face distance: " + faceDistString + ". Closest Distance: " + minDistString);
                    if(numAttempts > MAX_ATTEMPTS)
                        finishDetection("Unknown face", false);
                } else if (minDist < distanceThreshold) { // 3. Distant from face space and near a face class
                    //Log.d(TAG, "False recognition. Face distance: " + faceDistString + ". Closest Distance: " + minDistString);
                    if(numAttempts > MAX_ATTEMPTS)
                        finishDetection("False recognition", false);
                } else {// 4. Distant from face space and not near a known face class.
                    //Log.d(TAG, "Image is not a face. Face distance: " + faceDistString + ". Closest Distance: " + minDistString);
                    if(numAttempts > MAX_ATTEMPTS)
                        finishDetection("False face", false);
                }
            }
        }
    };

    private void finishDetection(String result, boolean ok) {
        this.processFrames=false;
        this.detectingFace=false;
        this.detectingDocumentFront=false;
        this.detectingDocumentBack=false;
        sendFinalResult(result, ok);
    }

    private void sendFinalResult(String result, boolean ok) {
        final PluginResult.Status status = ok ? PluginResult.Status.OK : PluginResult.Status.ERROR;

        PluginResult pluginResult=null;

        try {
            JSONObject json = new JSONObject();
            json.put("result", result);
            pluginResult = new PluginResult(status, json);
        } catch (JSONException e) {
            e.printStackTrace();
            pluginResult = new PluginResult(status);
        }

        pluginResult.setKeepCallback(false);
        cb.sendPluginResult(pluginResult);
    }
}
