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

import com.googlecode.tesseract.android.TessBaseAPI;
import com.legalpin.facelib.NativeMethods;

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
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
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
import java.util.List;
import java.util.function.Predicate;

import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_MEAN_C;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;

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
    //private Mat                  mYuv;
    //private Mat                  mYuvOrig;
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
    //private boolean detecting = false;

    private String currentTrainingFace = null;

    private int numAttempts = 0;

    private static final double FACE_THRESHOLD     = 0.033;
    private static final double DISTANCE_THRESHOLD = 0.007;
    private static final int NUM_USER_FACES        = 10;
    private static final int MAX_ATTEMPTS          = 10;

    private static final int C_FACE_DETECT     = 3;
    private static final int C_FACE_RECOG      = 4;
    private static final int C_FACE_TRAIN      = 5;
    private static final int C_ID_FRONT_DETECT = 6;
    private static final int C_ID_MRZ_DETECT   = 7;
    private static final int C_ID_AUTO_DETECT  = 8;

    private int detectorEngineState = 0;


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

            File tempDir = activity.getApplicationContext().getCacheDir();

            File tessDir = new File(tempDir.getAbsolutePath()+"/tessdata");
            if(!tessDir.exists())
                tessDir.mkdir();
            else {
                File tessFile = new File(tempDir.getAbsolutePath()+"/tessdata/eng.traineddata");
                if(!tessFile.exists()) {
                    InputStream is = am.open("www/data/eng.traineddata");
                    OutputStream os = new FileOutputStream(tessFile);

                    byte[] buffer = new byte[8 * 1024];
                    int bytesRead;
                    while ((bytesRead = is.read(buffer)) != -1)
                        os.write(buffer, 0, bytesRead);

                    os.close();
                    is.close();
                }
            }

        } catch (Exception e) {
            Log.e(TAG, "Error initializing: "+e.getClass().getName()+": "+e.getMessage());
            e.printStackTrace();
        }
    }
/*
    final static private boolean hasFlag(int state, int flag) {
        return (state & flag) != 0;
    }
*/
/*
    final static private int setFlag(int state, int flag) {
        return state | flag;
    }
*/
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
            this.detectorEngineState = 0;

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
            PluginResult result = new PluginResult(PluginResult.Status.OK, detectorEngineState==C_FACE_TRAIN);
            result.setKeepCallback(false);
            callbackContext.sendPluginResult(result);

            return true;
        }

        if(action.equals("startDetecting")) {
            cb = callbackContext;

            Log.d(TAG, "startDetecting(): called");

            int type = -1;
            try {
                type = data.getInt(0);
            } catch (JSONException je) {
            }

            if(type==C_FACE_TRAIN) {
                Log.e(TAG, "Face training cannot be called from startDetecting");
                this.sendFinalResult("Face training cannot be called from startDetecting", false);

                return true;
            }

            Log.d(TAG, "startDetecting: type="+type);
            if (type==-1 || type == C_FACE_RECOG)
                startRecognizingFace();
            else if (type == C_ID_FRONT_DETECT)
                startDetectingDocumentFront();
            else if (type == C_ID_MRZ_DETECT)
                startDetectingDocumentMRZ();
            else if (type == C_FACE_DETECT)
                startDetectingFaceSimple();
            else if (type == C_ID_AUTO_DETECT)
                startDetectingDocumentAuto();
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
            this.detectorEngineState = 0;

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
        detectorEngineState = C_FACE_TRAIN;
        this.processFrames = true;
    }

    private boolean faceExists(String faceName) {
        for(String face: facesList)
            if(face.equalsIgnoreCase(faceName))
                return true;

        return false;
    }

    private void startRecognizingFace() {
        Log.d(TAG, "startRecognizingFace(): called");

        if(facesList.size()==0) {
            cb.error("NO FACES");
            return;
        }

        PluginResult pluginResult = new  PluginResult(PluginResult.Status.NO_RESULT);
        pluginResult.setKeepCallback(true); // Keep callback
        cb.sendPluginResult(pluginResult);


        this.numAttempts = 0;
        this.processFrames = true;
        detectorEngineState = C_FACE_RECOG;
    }

    private void startDetectingDocumentFront() {
        Log.d(TAG, "startRecognizingDocumentFront(): called");

        PluginResult pluginResult = new  PluginResult(PluginResult.Status.NO_RESULT);
        pluginResult.setKeepCallback(true); // Keep callback
        cb.sendPluginResult(pluginResult);
        this.processFrames = true;
        detectorEngineState = C_ID_FRONT_DETECT;
    }

    private void startDetectingDocumentMRZ() {
        Log.d(TAG, "startRecognizingDocumentMRZ(): called");

        PluginResult pluginResult = new  PluginResult(PluginResult.Status.NO_RESULT);
        pluginResult.setKeepCallback(true); // Keep callback
        cb.sendPluginResult(pluginResult);
        this.processFrames = true;
        detectorEngineState = C_ID_MRZ_DETECT;
    }

    private void startDetectingDocumentAuto() {
        Log.d(TAG, "startDetectingDocumentAuto(): called");

        PluginResult pluginResult = new  PluginResult(PluginResult.Status.NO_RESULT);
        pluginResult.setKeepCallback(true); // Keep callback
        cb.sendPluginResult(pluginResult);

        this.processFrames = true;
        detectorEngineState = C_ID_AUTO_DETECT;
    }


    private void startDetectingFaceSimple() {
        Log.d(TAG, "startDetectingFaceSimple(): called");

        PluginResult pluginResult = new  PluginResult(PluginResult.Status.NO_RESULT);
        pluginResult.setKeepCallback(true); // Keep callback
        cb.sendPluginResult(pluginResult);

        this.numAttempts = 0;
        this.processFrames = true;
        detectorEngineState = C_FACE_DETECT;
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
        detectorEngineState=0;

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

                    Mat mYuvOrig = new Mat(height, width, CvType.CV_8UC1);
                    Mat mYuv = matDup(mYuvOrig);
                    mYuvOrig.put(0, 0, data);

                    double sharpness = getSharpness(mYuvOrig);
                    if(sharpness>30) {
                        //saveImgSD("test1.jpeg", mYuvOrig);
                        if (detectorEngineState == C_FACE_RECOG) {
                            rotateToPortrait(mYuvOrig, mYuv);
                            if (detectFace(mYuv, data, sharpness))
                                recognizeFace(mYuv);
                        } else if (detectorEngineState == C_FACE_TRAIN) {
                            rotateToPortrait(mYuvOrig, mYuv);
                            if (detectFace(mYuv, data, sharpness))
                                storeFace(mYuv);
                        } else if (detectorEngineState == C_ID_FRONT_DETECT) {
                            mYuv = mYuvOrig;
                            detectDocumentFront(mYuv, sharpness);
                        } else if (detectorEngineState == C_ID_MRZ_DETECT) {
                            mYuv = mYuvOrig;
                            detectDocumentMRZ(mYuv, data, sharpness);
                        } else if (detectorEngineState == C_ID_AUTO_DETECT) {
                            mYuv = mYuvOrig;
                            Object[] detectResult = detectDocumentAuto(mYuv, sharpness);
                            if(detectResult!=null)
                                sendDocument(height, width, data, detectResult);
                        } else if (detectorEngineState == C_FACE_DETECT) {
                            rotateToPortrait(mYuvOrig, mYuv);
                            //saveImgSD("test2.jpeg", mYuv);
                            if (detectFace(mYuv, data, sharpness))
                                sendFace(height, width, data, sharpness);
                        }
                    } else{
                        Log.d(TAG, "Blurry image: "+sharpness);
                    }

                    thread_over = true;
                }
                //update time and reset timeout
                last_time = current_time;
            }

        }
    };

    private void rotateToPortrait(Mat mYuvSrc, Mat mYuvDst) {
        Core.transpose(mYuvSrc, mYuvDst);
        Core.flip(mYuvDst, mYuvDst, -1);
    }

    private boolean detectFace(Mat mYuv, byte[] data, double sharpness) {
        int width = mYuv.cols();
        int height = mYuv.rows();

        int minDimension = height > width ? width : height;

        final MatOfRect faces = new MatOfRect();
        final int absoluteFaceSize = Math.round(minDimension * 0.5f);
        final int minValidFaceSize = Math.round(minDimension * 0.65f);

        //faceCascade.detectMultiScale(mYuv, faces, 1.3, 2, Objdetect.CASCADE_SCALE_IMAGE, new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        faceCascade.detectMultiScale(mYuv, faces, 1.3, 5, Objdetect.CASCADE_SCALE_IMAGE, new Size(absoluteFaceSize, absoluteFaceSize), new Size());

        org.opencv.core.Rect[] facesArray = faces.toArray();

        final int facesLength = facesArray.length;
        if (facesLength == 1 && facesArray[0].height >= minValidFaceSize) {
            org.opencv.core.Rect face = facesArray[0];
            Log.d(TAG,"FACE CASCADE CLASSIFIER OK!: IH:"+mYuv.height()+"AFS:" +absoluteFaceSize+" H:"+face.height+" W:"+face.width);
            return true;
        }

        return false;

    }

    private void storeFace(Mat image_pattern) {
        try {
            int numFace = imagesFacesTmp.size();
            Log.d(TAG, "Storing face #" + imagesFacesTmp.size());
            //String extStorageDirectory = Environment.getExternalStorageDirectory().toString();
            //File lpinDir = new File(extStorageDirectory + "/lpin");
            //lpinDir.mkdir();
            //Imgcodecs.imwrite(extStorageDirectory + "/lpin/pic_" + numFace + ".png", image_pattern);

            Mat image = image_pattern.reshape(0, (int) image_pattern.total()); // Create column vector


            if (imagesFacesTmp.size() >= NUM_USER_FACES) {
                this.detectorEngineState = 0;
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
    private void sendDocument(int height, int width, byte[] data, Object[] detectResult) {
        Mat mYuv = new Mat(height+height/2, width, CvType.CV_8UC1);
        mYuv.put(0,0, data);
        Mat mRgba = new Mat(height, width, CvType.CV_8UC4);
        Imgproc.cvtColor(mYuv, mRgba, Imgproc.COLOR_YUV420sp2BGRA, 4 );

        JSONObject json = (JSONObject) detectResult[0];
        MatOfPoint2f pespective = (MatOfPoint2f) detectResult[1];
        Mat mFinal = applyTransformationRotation(mRgba, pespective);

        String documentPath = this.saveImgTemp("documentFront.jpeg", mFinal);

        try {
            json.put("result", documentPath);
        } catch (JSONException e) {
            e.printStackTrace();
        }

        finishDetection(json, true);
    }

    private void sendFace(int height, int width, byte[] data, double sharpness) {
        Mat mYuv = new Mat(height+height/2, width, CvType.CV_8UC1);
        mYuv.put(0,0, data);
        Mat mRgba = new Mat(height, width, CvType.CV_8UC4);

        Imgproc.cvtColor(mYuv, mRgba, Imgproc.COLOR_YUV420sp2BGRA, 4 );
        rotateToPortrait(mRgba, mRgba);
        String facePath = this.saveImgTemp("returnFace.jpeg", mRgba);

        JSONObject json = null;
        try {
            json = new JSONObject();
            json.put("result", facePath);
            json.put("sharpness", sharpness);
        } catch (JSONException e) {
            e.printStackTrace();
            json = null;
        }

        finishDetection(json, true);
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


    public void recognizeFace(Mat mGray) {
        Log.d(TAG, "recognizeFace() called");
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



    public Object[] detectDocumentGeneric(Mat mGrayInput, double sharpness) {
/*
        Mat mGray = matDup(mGrayInput);
        Imgproc.equalizeHist(mGrayInput, mGray);
*/
        Mat mGray = mGrayInput;
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

        //double sharpness = getSharpness(mGray);
        Log.d(TAG, "Image sharpness: "+sharpness);

        if(sharpness < 80) {
            Log.d(TAG, "Blurry image not valid");
            return null;
        }

        MatOfPoint2f contoursSmall = findContours(mSmall);
        if(contoursSmall == null) {
            Log.d(TAG, "No contours found");
            return null;
        }

        MatOfPoint2f contoursOK = correctPerspective(contoursSmall, mSmall.size());
/*
        if(contoursOK.get(0,0)[0]!=contoursSmall.get(0,0)[0] ||
                contoursOK.get(0,0)[1]!=contoursSmall.get(0,0)[1])
            Log.d(TAG, " * PERSPECTIVE CORRECTION contoursSmall: "+getContourToString(contoursSmall)+" contoursOK: "+getContourToString(contoursOK));
*/
        MatOfPoint2f bigContours = recalculateContours(mSmall.size(), mGray.size(), contoursOK);

        //Log.d(TAG, "detectDocumentFront(): rectangle detected");

        Mat mWarpedSmall = applyTransformationRotation(mSmall, contoursOK);
        if(mWarpedSmall==null) {
            Log.d(TAG, "No contours found!");
            return null;
        }

        Mat mWarpedBig = applyTransformationRotation(mGray, bigContours);

        return new Mat[]{mWarpedBig, mWarpedSmall, bigContours};
    }


    public void detectDocumentFront(Mat mGray, double sharpness) {
        Log.d(TAG, "detectDocumentFront() called");

        Object[] dataArray = this.detectDocumentGeneric(mGray, sharpness);

        if(dataArray==null)
            return;

        final Mat mWarpedBig   = (Mat) dataArray[0];

        String path = this.saveImgTemp("documentFront.jpeg", mWarpedBig);

        this.finishDetection(path, true);
    }


    private String processMRZPart(Mat mWarpedBig, Mat mWarpedSmall, Rect rectSmall) {
        Rect rectBig = recalculateRect(mWarpedSmall.size(), mWarpedBig.size(), rectSmall);
        return getImageToString(mWarpedBig, rectBig);
    }

    public void detectDocumentMRZ(Mat mGray, byte[] data, double sharpness) {
        Log.d(TAG, "detectDocumentMRZ() called");

        final Object[] dataArray = this.detectDocumentGeneric(mGray, sharpness);

        if(dataArray==null)
            return;

        final Mat mWarpedBig   = (Mat) dataArray[0];
        final Mat mWarpedSmall = (Mat) dataArray[1];

        Rect rectSmall = detectMRZ(mWarpedSmall);
        if (rectSmall == null)
            return;

        String text = processMRZPart(mWarpedBig, mWarpedSmall, rectSmall);

        Log.d(TAG, "TEXTO EXTRAIDO:\n" + text);
        if(text==null)
            return;

        String path = this.saveImgTemp("documentFront.jpeg", mWarpedBig);
        JSONObject json = null;
        try {
            json = new JSONObject();
            json.put("result", path);
            json.put("text",text);
        } catch (JSONException e) {
            //e.printStackTrace();
            json = null;
        }

        if(json==null)
            return;

        this.finishDetection(json, true);
    }

    public Object[] detectDocumentAuto(Mat mGray, double sharpness) {
        Log.d(TAG, "detectDocumentAuto() called");

        final Object[] data = this.detectDocumentGeneric(mGray, sharpness);

        if(data==null)
            return null;

        final Mat mWarpedBig   = (Mat) data[0];
        final Mat mWarpedSmall = (Mat) data[1];
        final MatOfPoint2f contours = (MatOfPoint2f) data[2];

        Rect rectSmall = detectMRZ(mWarpedSmall);
        boolean hasMrz = rectSmall != null;

        String text = null;

        if(hasMrz)
            text = processMRZPart(mWarpedBig, mWarpedSmall, rectSmall);

        Log.d(TAG, "TEXTO EXTRAIDO:\n" + text);

        JSONObject json = null;
        try {
            json = new JSONObject();
            json.put("mrz", hasMrz ? true : false);
            json.put("sharpness", sharpness);
            if(hasMrz && text!=null)
                json.put("text", text);
        } catch (JSONException e) {
            //e.printStackTrace();
            json = null;
        }

        if(json==null)
            return null;

        return new Object[]{json, contours};
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
        final Mat image_roi = new Mat(image, rect);

        final String imagePath = this.saveImgTemp("mrz_text.jpeg", image_roi);
        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1; // 1 - means max size. 4 - means maxsize/4 size. Don't use value <4, because you need more memory in the heap to store your data.
        final Bitmap bitmap = BitmapFactory.decodeFile(imagePath, options);

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

        Environment.getDataDirectory();
        tessBaseApi.init(activity.getApplicationContext().getCacheDir().getAbsolutePath(), "eng");
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

    @SuppressLint("NewApi")
    private Rect detectMRZ(Mat image) {
        //saveImgSD("test2.jpeg",image);
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
                //saveImgSD("test3.jpeg", imageRect);
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

    private void saveImgX(String path, Mat img) {
        Log.d(TAG, "Saving img: "+path);
        Imgcodecs.imwrite(path, img);
    }

    private String saveImgSD(String name, Mat img) {
        File rootDir = Environment.getExternalStorageDirectory();
        String path = rootDir.getAbsolutePath() + "/" + name;
        saveImgX(rootDir.getAbsolutePath() + "/" + name, img);
        return path;
    }

    final private File getCacheDir() {
        return activity.getApplicationContext().getCacheDir();
    }

    private String saveImgTemp(String name, Mat img) {
        File rootDir = getCacheDir();
        String path = rootDir.getAbsolutePath() + "/" + name;
        saveImgX(rootDir.getAbsolutePath() + "/" + name, img);
        return path;
    }

    private Mat getEdgesFromImage(Mat mGray) {
        //saveImgSD("x1.jpeg", mGray);
        Mat mThres = matDup(mGray);

        //Imgproc.adaptiveThreshold(mGray, mThres, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 40);

        //saveImgSD("x2.jpeg", mThres);

        Mat mGaussian = new Mat(mGray.rows(),mGray.cols(),mGray.type());


        Mat mCanny = mGaussian.clone();

        //Core.multiply(mGray, new Scalar(2.0), mMul);
        //Core.add(mMul, new Scalar(50.0), mAdd);

        //Imgproc.GaussianBlur(mAdd, mGaussian, new Size(5,5), 0);
        //Imgproc.GaussianBlur(mMul, mGaussian, new Size(5,5), 0);
        Imgproc.GaussianBlur(mGray, mGaussian, new Size(5,5), 0);
        Imgproc.Canny(mGaussian, mCanny, 75, 200);

        return mCanny;
    }

    private Mat matDup(Mat src) {
        return new Mat(src.cols(), src.rows(), src.type());
    }

    MatOfPoint getMatOfPointFromInt(MatOfPoint mopIn, MatOfInt hull) {
        MatOfPoint mopOut = new MatOfPoint();
        mopOut.create((int)hull.size().height,1,CvType.CV_32SC2);

        for(int i = 0; i < hull.size().height ; i++)
        {
            int index = (int)hull.get(i, 0)[0];
            double[] point = new double[] {
                    mopIn.get(index, 0)[0], mopIn.get(index, 0)[1]
            };
            mopOut.put(i, 0, point);
        }

        return mopOut;
    }

    @SuppressLint("NewApi")
    private MatOfPoint2f findContours(Mat mImage) {
        Mat mCanny = getEdgesFromImage(mImage);

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

        if(contours.size()>5)
            contours = contours.subList(0, 5);

        for(MatOfPoint c : contours) {
            //Minimun size allowed for consideration
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            MatOfPoint2f contour2f = new MatOfPoint2f(c.toArray());


            //Processing on mMOP2f1 which is in type MatOfPoint2f

            //double approxDistance = Imgproc.arcLength(contour2f,true)*0.02;
            double approxDistance = Imgproc.arcLength(contour2f,true)*0.05;
            Imgproc.approxPolyDP(contour2f,approxCurve,approxDistance,true);

            //convert to MatofPoint
            if(approxCurve.toArray().length == 4) {
                MatOfPoint approxCurvePoint = new MatOfPoint(approxCurve.toArray());
/*
                final Mat mImgTest = mImage.clone();

                Rect r1 = Imgproc.boundingRect(approxCurvePoint);

                Imgproc.rectangle(mImgTest, r1.tl(), r1.br(),new Scalar(255), 2);
                saveImgSD("test1.jpeg", mImgTest);
*/
                return approxCurve;
            }

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
        this.processFrames = false;
        this.detectorEngineState = 0;
        sendFinalResult(result, ok);
    }

    private void finishDetection(JSONObject json, boolean ok) {
        this.processFrames = false;
        this.detectorEngineState = 0;
        sendFinalResult(json, ok);
    }

    private void sendFinalResult(String result, boolean ok) {
        JSONObject json = null;
        try {
            json = new JSONObject();
            json.put("result", result);
        } catch (JSONException e) {
            e.printStackTrace();
            json = null;
        }

        this.sendFinalResult(json, ok);
    }

    private void sendFinalResult(JSONObject json, boolean ok) {
        final PluginResult.Status status = ok ? PluginResult.Status.OK : PluginResult.Status.ERROR;

        final PluginResult pluginResult;
        if (json != null)
            pluginResult = new PluginResult(status, json);
        else
            pluginResult = new PluginResult(status);

        pluginResult.setKeepCallback(false);
        cb.sendPluginResult(pluginResult);
    }
}
