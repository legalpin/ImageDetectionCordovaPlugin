/**
 *
 **/
var ImageDetectionPlugin = function () {};

ImageDetectionPlugin.prototype.DETECT_FACE = 1;
ImageDetectionPlugin.prototype.DETECT_ID_FRONT = 2;
ImageDetectionPlugin.prototype.DETECT_ID_BACK = 3;
ImageDetectionPlugin.prototype.DETECT_FACE_SIMPLE = 4;

ImageDetectionPlugin.prototype.CAMERA_FRONT = 1;
ImageDetectionPlugin.prototype.CAMERA_BACK = 2;


ImageDetectionPlugin.prototype.openCamera = function (type, successCallback, errorCallback) {
  cordova.exec(successCallback, errorCallback, "ImageDetectionPlugin", "openCamera", [type]);
};
ImageDetectionPlugin.prototype.closeCamera = function (successCallback, errorCallback) {
  cordova.exec(successCallback, errorCallback, "ImageDetectionPlugin", "closeCamera", []);
};
ImageDetectionPlugin.prototype.isDetecting = function (successCallback, errorCallback) {
  cordova.exec(successCallback, errorCallback, "ImageDetectionPlugin", "isDetecting", []);
};
ImageDetectionPlugin.prototype.setDetectionTimeout = function (timeout, successCallback, errorCallback) {
  cordova.exec(successCallback, errorCallback, "ImageDetectionPlugin", "setDetectionTimeout", [timeout]);
};
ImageDetectionPlugin.prototype.startTraining = function (faceName, successCallback, errorCallback) {
  cordova.exec(successCallback, errorCallback, "ImageDetectionPlugin", "startTraining", [faceName]);
};
ImageDetectionPlugin.prototype.startDetecting = function (type, successCallback, errorCallback) {
  cordova.exec(successCallback, errorCallback, "ImageDetectionPlugin", "startDetecting", [type]);
};
ImageDetectionPlugin.prototype.stopDetecting = function (successCallback, errorCallback) {
  cordova.exec(successCallback, errorCallback, "ImageDetectionPlugin", "stopDetecting", []);
};

if (!window.plugins) {
  window.plugins = {};
}

if (!window.plugins.ImageDetectionPlugin) {
  window.plugins.ImageDetectionPlugin = new ImageDetectionPlugin();
}

if (typeof module != 'undefined' && module.exports){
  module.exports = ImageDetectionPlugin;
}
