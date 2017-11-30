package com.legalpin.imagedetectionplugin;

import android.Manifest;
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
import android.graphics.Rect;
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
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfRect;
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
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import com.legalpin.facelib.NativeMethods;

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
    private boolean useEigenfaces = true;
    private ArrayList<String> imagesLabels;
    private String[] uniqueLabels;
    private boolean training;
    private boolean detecting;

    private int numFaces=0;
    final private int TOTAL_FACES = 5;

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

        if (action.equals("openCamera")) {
            cb = callbackContext;
            this.processFrames = false;
            this.training = false;
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
                Log.e(TAG, " ********************************** FACE NAME ADQUISITION ERROR");
                je.printStackTrace();
            }

            if(faceName==null || "".equals(faceName)) {
                Log.e(TAG, " ********************************** No face name");
                //PluginResult result = new PluginResult(PluginResult.Status.ERROR);
                //result.setKeepCallback(false);
                //cb.sendPluginResult(result);
                cb.error("FAIL2");
                return true;
            }

            imagesLabels = new ArrayList<String>();
            imagesLabels.add(faceName);
            uniqueLabels = imagesLabels.toArray(new String[0]);
            PluginResult pluginResult = new  PluginResult(PluginResult.Status.NO_RESULT);
            pluginResult.setKeepCallback(true); // Keep callback
            cb.sendPluginResult(pluginResult);

            startTraining();

            return true;
        }
/*
        if (action.equals("isDetecting")) {
            Log.i(TAG, "isDetecting called");

            this.detecting=true;
            PluginResult result = new PluginResult(PluginResult.Status.OK, this.processFrames);
            result.setKeepCallback(false);
            callbackContext.sendPluginResult(result);

            return true;
        }
*/
        if (action.equals("isTraining")) {
            Log.i(TAG, "isTraining called");
            PluginResult result = new PluginResult(PluginResult.Status.OK, this.training);
            result.setKeepCallback(false);
            callbackContext.sendPluginResult(result);

            return true;
        }

        if(action.equals("startDetecting")) {
            Log.i(TAG, "startDetecting called");
            this.processFrames = true;
            this.detecting=true;
            return true;
        }

        if(action.equals("stopDetecting")) {
            Log.i(TAG, "stopDetecting called");
            this.processFrames = false;
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

    private void startTraining() {
        Log.d(TAG, "startTraining(): Starting engaged!!!!!");
        if(imagesFaces==null)
            imagesFaces = new ArrayList<Mat>();
        this.numFaces = 0;
        this.detecting = false;
        this.training = true;
        this.processFrames = true;
    }

    private void startDetecting() {
        Log.d(TAG, "startDetecting(): Starting engaged!!!!!");
        imagesFaces = new ArrayList<Mat>();
        this.numFaces = 0;
        this.training = false;
        this.detecting = true;
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
                    thread_over = false;                    //Log.d(TAG, "Processing frame inner: "+System.currentTimeMillis());
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

                    final int facesLength = facesArray.length;
                    if (facesLength == 1) {
                        if(training) {
                            saveFace(mYuv);
                            //Log.d(TAG, "Faces array size: " + facesLength);
                        }

                        if(detecting) {
                            detectFace(mYuv);
                            //Log.d(TAG, "Faces array size: " + facesLength);
                        }


                    }
/*
                    if(!training) {
                        cordova.getThreadPool().execute(new Runnable() {
                            public void run() {
                                try {
                                    JSONObject json = new JSONObject();
                                    json.put("faces", facesLength);
                                    PluginResult result = new PluginResult(PluginResult.Status.OK, json);
                                    result.setKeepCallback(true);
                                    cb.sendPluginResult(result);
                                } catch (JSONException e) {
                                    e.printStackTrace();
                                    ;
                                }
                            }
                        });
                    }
*/
                    thread_over = true;
                }
                //update time and reset timeout
                last_time = current_time;
            }

        }
    };

    private void saveFace(Mat image_pattern) {
        try {
            Log.d(TAG, "Saving face #" + numFaces);
            String extStorageDirectory = Environment.getExternalStorageDirectory().toString();
            File lpinDir = new File(extStorageDirectory + "/lpin");
            lpinDir.mkdir();
            Imgcodecs.imwrite(extStorageDirectory + "/lpin/pic_" + this.numFaces + ".png", image_pattern);

            Mat image = image_pattern.reshape(0, (int) image_pattern.total()); // Create column vector
            imagesFaces.add(image);
            if (numFaces > TOTAL_FACES) {
                training = false;
                trainFaces();
            } else {
                JSONObject json = new JSONObject();
                json.put("numFaces", numFaces);
                json.put("totalFaces", TOTAL_FACES);

                PluginResult result = new PluginResult(PluginResult.Status.OK, json);
                result.setKeepCallback(true);
                cb.sendPluginResult(result);
            }
            numFaces++;
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
        Log.d(TAG, " ******************************************************* trainFaces(): CALLED");
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

            Set<String> uniqueLabelsSet = new HashSet<>(imagesLabels); // Get all unique labels
            uniqueLabels = uniqueLabelsSet.toArray(new String[uniqueLabelsSet.size()]); // Convert to String array, so we can read the values from the indices

            int[] classesNumbers = new int[uniqueLabels.length];
            for (int i = 0; i < classesNumbers.length; i++)
                classesNumbers[i] = i + 1; // Create incrementing list for each unique label starting at 1

            int[] classes = new int[imagesLabels.size()];
            for (int i = 0; i < imagesLabels.size(); i++) {
                String label = imagesLabels.get(i);
                for (int j = 0; j < uniqueLabels.length; j++) {
                    if (label.equals(uniqueLabels[j])) {
                        classes[i] = classesNumbers[j]; // Insert corresponding number
                        break;
                    }
                }
            }

            /*for (int i = 0; i < imagesLabels.size(); i++)
                Log.i(TAG, "Classes: " + imagesLabels.get(i) + " = " + classes[i]);*/

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
        Log.d(TAG, " ************************************************ detectFace() called!!!!!!!!!!!!!");
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
/*
        images.add(image); // Add current image to the array

        if (images.size() > maximumImages) {
            images.remove(0); // Remove first image
            imagesLabels.remove(0); // Remove first label
            Log.i(TAG, "The number of images is limited to: " + images.size());
        }
*/
        // Calculate normalized Euclidean distance
        mMeasureDistTask = new NativeMethods.MeasureDistTask(useEigenfaces, measureDistTaskCallback);
        mMeasureDistTask.execute(image);
        //mMeasureDistTask.execute(mGray);

        //showLabelsDialog();
    }


    private NativeMethods.MeasureDistTask.Callback measureDistTaskCallback = new NativeMethods.MeasureDistTask.Callback() {
        @Override
        public void onMeasureDistComplete(Bundle bundle) {
            if (bundle == null) {
                Log.d(TAG, "Failed to measure distance: "+ Toast.LENGTH_LONG);
                return;
            }

            Log.d(TAG, "Measure distance callback: "+ bundle.toString());
            double faceThreshold = 0.020;
            double distanceThreshold = 0.020;


            float minDist = bundle.getFloat(NativeMethods.MeasureDistTask.MIN_DIST_FLOAT);
            if (minDist != -1) {
                int minIndex = bundle.getInt(NativeMethods.MeasureDistTask.MIN_DIST_INDEX_INT);
                float faceDist = bundle.getFloat(NativeMethods.MeasureDistTask.DIST_FACE_FLOAT);
                if (imagesLabels.size() > minIndex) { // Just to be sure
                    Log.i(TAG, "dist[" + minIndex + "]: " + minDist + ", face dist: " + faceDist + ", label: " + imagesLabels.get(minIndex));

                    String minDistString = String.format(Locale.US, "%.4f", minDist);
                    String faceDistString = String.format(Locale.US, "%.4f", faceDist);

                    if (faceDist < faceThreshold && minDist < distanceThreshold) {// 1. Near face space and near a face class
                        Log.d(TAG, "Face detected: " + imagesLabels.get(minIndex) + ". Distance: " + minDistString);
                        sendFinalResult("Face detected");
                    } else if (faceDist < faceThreshold) { // 2. Near face space but not near a known face class
                        Log.d(TAG, "Unknown face. Face distance: " + faceDistString + ". Closest Distance: " + minDistString);
                        sendFinalResult("Unknown face");
                    } else if (minDist < distanceThreshold) { // 3. Distant from face space and near a face class
                        Log.d(TAG, "False recognition. Face distance: " + faceDistString + ". Closest Distance: " + minDistString);
                        sendFinalResult("False recognition");
                    } else {// 4. Distant from face space and not near a known face class.
                        Log.d(TAG, "Image is not a face. Face distance: " + faceDistString + ". Closest Distance: " + minDistString);
                        sendFinalResult("False face");
                    }
                }
            } else {
                Log.w(TAG, "Array is null");
                if (useEigenfaces || uniqueLabels == null || uniqueLabels.length > 1)
                    Log.d(TAG, "Keep training...");
                else
                    Log.d(TAG, "Fisherfaces needs two different faces");
            }
        }
    };

    private void sendFinalResult(String result) {
        PluginResult pluginResult = null;
        try {
            JSONObject json = new JSONObject();
            json.put("result", result);
            pluginResult = new PluginResult(PluginResult.Status.OK, json);
        } catch (JSONException e) {
            e.printStackTrace();
            pluginResult = new PluginResult(PluginResult.Status.OK);
        }

        pluginResult.setKeepCallback(false);
        cb.sendPluginResult(pluginResult);
    }
}
