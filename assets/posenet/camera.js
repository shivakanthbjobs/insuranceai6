const maxVideoSize = 600;
const canvasSize = 600;

async function getCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter(({kind}) => kind === 'videoinput');
}

let currentStream = null;

function stopWebCamOnly(camContainer) {
  if (currentStream) {
    currentStream.getTracks().forEach((track) => {
      track.stop();
    });
  }
}

function loadVideo(cameraId,camName,camContainer){
  return new Promise((resolve, reject) => {
    stopWebCamOnly(camContainer);

    const video = document.getElementById(camName);

    video.width = maxVideoSize;
    video.height = maxVideoSize;

    if (navigator.getUserMedia) {
      navigator.getUserMedia({
        video: {
          width: maxVideoSize,
          height: maxVideoSize,
          deviceId: {exact: cameraId},
        },
      }, handleVideo, videoError);
    }

    function handleVideo(stream) {
      currentStream = stream;
      video.srcObject = stream;

      resolve(video);
    }

    function videoError(e) {
      // do something
      reject(e);
    }
  });
}

const guiState = {
  algorithm: 'single-pose',
  input: {
    mobileNetArchitecture: '1.01',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 2,
    minPoseConfidence: 0.1,
    minPartConfidence: 0.3,
    nmsRadius: 20.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
  },
  net: null,
};

function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const cameraOptions = cameras.reduce((result, {label, deviceId}) => {
    result[label] = deviceId;
    return result;
  }, {});

}



function detectPoseInRealTime(video, net,camOutput){
  const canvas = document.getElementById(camOutput);
  const ctx = canvas.getContext('2d');
  const flipHorizontal = true;

  canvas.width = canvasSize;
  canvas.height = canvasSize;

  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      guiState.net.dispose();

      guiState.net = await posenet.load(Number(guiState.changeToArchitecture));

      guiState.changeToArchitecture = null;
    }

    

    const imageScaleFactor = guiState.input.imageScaleFactor;
    const outputStride = Number(guiState.input.outputStride);

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;
    switch (guiState.algorithm) {
    case 'single-pose':
      const pose = await guiState.net.estimateSinglePose(video, imageScaleFactor, flipHorizontal, outputStride);
      poses.push(pose);

      minPoseConfidence = Number(
        guiState.singlePoseDetection.minPoseConfidence);
      minPartConfidence = Number(
        guiState.singlePoseDetection.minPartConfidence);
      break;
    case 'multi-pose':
      poses = await guiState.net.estimateMultiplePoses(video, imageScaleFactor, flipHorizontal, outputStride,
        guiState.multiPoseDetection.maxPoseDetections,
        guiState.multiPoseDetection.minPartConfidence,
        guiState.multiPoseDetection.nmsRadius);

      minPoseConfidence = Number(guiState.multiPoseDetection.minPoseConfidence);
      minPartConfidence = Number(guiState.multiPoseDetection.minPartConfidence);
      break;
    }

    ctx.clearRect(0, 0, canvasSize, canvasSize);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvasSize*-1, canvasSize);
      ctx.restore();
    }

    const scale = canvasSize / video.width;

    if(camOutput=="posenetOutput") {
        poses.forEach(({score, keypoints}) => {
            if (score >= minPoseConfidence) {
              if (guiState.output.showPoints) {
                drawKeypoints(keypoints, minPartConfidence, ctx, scale);
              }
              if (guiState.output.showSkeleton) {
                drawSkeleton(keypoints, minPartConfidence, ctx, scale);
              }
            }
          });

    }
    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

async function startWebCamOnly(camName,camOutput,camContainer) {
  const net = await posenet.load();

  
  document.getElementById('loading').style.display = 'none';
  document.getElementById(camContainer).style.display = 'block';
//  document.getElementById('loading').style.display = 'none';
  

  const cameras = await getCameras();

  if (cameras.length === 0) {
    alert('No webcams available.  Reload the page when a webcam is available.');
    return;
  }

  const video = await loadVideo(cameras[0].deviceId,camName,camContainer);

  setupGui(cameras, net);
  //setupFPS();
  detectPoseInRealTime(video, net,camOutput);
  
}

navigator.getUserMedia = navigator.getUserMedia ||
navigator.webkitGetUserMedia ||
navigator.mozGetUserMedia;
