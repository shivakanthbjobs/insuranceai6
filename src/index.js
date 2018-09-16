import * as tf from '@tensorflow/tfjs';
import {Webcam} from './webcam';

import * as posenet from '@tensorflow-models/posenet';
import dat from 'dat.gui';
import Stats from 'stats.js';
import {drawKeypoints, drawSkeleton, drawBoundingBox} from './demo_util';


const faceapi = require("face-api.js");
var $ = require("jquery");


const webcam = new Webcam(document.getElementById('webcam'));


let age_model;
let video_element;

let minConfidence = 0.6;
let modelLoaded = false;


async function init() {
    try {
        age_model = await tf.loadModel('./age_models/model.json')

        faceapi.loadFaceDetectionModel('./models')
        faceapi.loadFaceLandmarkModel('./models')

        await webcam.setup()

        video_element = $('#webcam').get(0)

        console.log('initialization completed')
        await bindPage()
        run()
    } catch (e) {
        console.log(e)
    }
}


async function run() {
    while (true) {
        const {width, height} = faceapi.getMediaDimensions(video_element)
        const input = await faceapi.toNetInput(video_element)

        const canvas = $('#overlay').get(0)
        canvas.width = width
        canvas.height = height

        const locations = await faceapi.locateFaces(input, minConfidence)
        var faceImages = await faceapi.extractFaces(input.inputs[0], locations)

        // detect landmarks and get the aligned face image bounding boxes
        const alignedFaceBoxes = await Promise.all(faceImages.map(
            async (faceCanvas, i) => {
                const faceLandmarks = await faceapi.detectLandmarks(faceCanvas)
                return faceLandmarks.align(locations[i])
            }
        ))
        faceImages = await faceapi.extractFaces(input.inputs[0], alignedFaceBoxes)

        // do things to each face
        const gender_tag = $('#gender')
        const age_tag = $('#age')
        faceImages.forEach(async (faceCanvas, i) => {
                const faceEl = $('#face')
                const ctx = canvas.getContext('2d')
                ctx.drawImage(faceCanvas, 0, 0)

                var img = tf.fromPixels(faceCanvas)
                const size = Math.min(img.shape[0], img.shape[1]);
                const centerHeight = img.shape[0] / 2;
                const beginHeight = centerHeight - (size / 2);
                const centerWidth = img.shape[1] / 2;
                const beginWidth = centerWidth - (size / 2);
                img = img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
                img = img.resizeBilinear([64, 64])
                img = img.expandDims(0);
                img = img.toFloat();
                //img = img.toFloat().div(tf.scalar(127)).sub(tf.scalar(1)); //Do not use this line 

                const results = age_model.predict(img)

                const predicted_genders = results[0].dataSync()
                if (predicted_genders[0] > 0.5) {
                    gender_tag.text("Female")
                    //console.log("Female")
                } else {
                    gender_tag.text("Male")
                    //console.log("Male")
                }
                
                const ages = tf.range(0, 101, 1).reshape([101, 1])
                const predicted_ages = results[1].dot(ages).flatten().dataSync()
                age_tag.text(predicted_ages[0] )
                //console.log("How old: ", predicted_ages[0] - 10)
        })

        faceapi.drawDetection('overlay', locations.map(det => det.forSize(width, height)))

        await tf.nextFrame()
    }
}


const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const guiState = {
  algorithm: 'multi-pose',
  input: {
    mobileNetArchitecture: isMobile() ? '0.50' : '0.75',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
  },
  net: null,
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({width: 300});

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  const algorithmController =
      gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);

  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  const architectureController = input.add(
      guiState.input, 'mobileNetArchitecture',
      ['1.01', '1.00', '0.75', '0.50']);
  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  input.add(guiState.input, 'outputStride', [8, 16, 32]);
  // Image scale factor: What to scale the image by before feeding it through
  // the network.
  input.add(guiState.input, 'imageScaleFactor').min(0.2).max(1.0);
  input.open();

  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  let single = gui.addFolder('Single Pose Detection');
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);

  let multi = gui.addFolder('Multi Pose Detection');
  multi.add(guiState.multiPoseDetection, 'maxPoseDetections')
      .min(1)
      .max(20)
      .step(1);
  multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0);
  multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0);
  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0);
  multi.open();

  let output = gui.addFolder('Output');
  output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.add(guiState.output, 'showBoundingBox');
  output.open();


  architectureController.onChange(function(architecture) {
    guiState.changeToArchitecture = architecture;
  });

  algorithmController.onChange(function(value) {
    switch (guiState.algorithm) {
      case 'single-pose':
        multi.close();
        single.open();
        break;
      case 'multi-pose':
        single.close();
        multi.open();
        break;
    }
  });
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');
  // since images are being fed from a webcam
  const flipHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();

      // Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01
      // version
      guiState.net = await posenet.load(+guiState.changeToArchitecture);

      guiState.changeToArchitecture = null;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    // Scale an image down to a certain factor. Too large of an image will slow
    // down the GPU
    const imageScaleFactor = guiState.input.imageScaleFactor;
    const outputStride = +guiState.input.outputStride;

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;
    switch (guiState.algorithm) {
      case 'single-pose':
        const pose = await guiState.net.estimateSinglePose(
            video, imageScaleFactor, flipHorizontal, outputStride);
        poses.push(pose);

        minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
        break;
      case 'multi-pose':
        poses = await guiState.net.estimateMultiplePoses(
            video, imageScaleFactor, flipHorizontal, outputStride,
            guiState.multiPoseDetection.maxPoseDetections,
            guiState.multiPoseDetection.minPartConfidence,
            guiState.multiPoseDetection.nmsRadius);

        minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
        break;
    }

    ctx.clearRect(0, 0, videoWidth, videoHeight);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }

    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores
    poses.forEach(({score, keypoints}) => {
      if (score >= minPoseConfidence) {
        if (guiState.output.showPoints) {
          drawKeypoints(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showSkeleton) {
          drawSkeleton(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showBoundingBox) {
          drawBoundingBox(keypoints, ctx);
        }
      }
    });

    // End monitoring code for frames per second
    stats.end();

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  // Load the PoseNet model weights with architecture 0.75
  const net = await posenet.load(0.75);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  setupGui([], net);
  setupFPS();
  detectPoseInRealTime(video, net);
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo




init()
//bindPage();
