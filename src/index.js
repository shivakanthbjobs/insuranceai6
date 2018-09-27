// import Artyom from '../node_modules/artyom.js/build/artyom';

// const Jarvis = new Artyom();

// Jarvis.say("Hello World !");

// // Create a variable that stores your instance
// const artyom = new Artyom();

// // Or if you are using it in the browser
// // var artyom = new Artyom();// or `new window.Artyom()`

// // Add command (Short code artisan way)
// artyom.on(['Good morning','Good afternoon']).then((i) => {
//     switch (i) {
//         case 0:
//             artyom.say("Good morning, how are you?");
//         break;
//         case 1:
//             artyom.say("Good afternoon, how are you?");
//         break;            
//     }
// });

// // Smart command (Short code artisan way), set the second parameter of .on to true
// artyom.on(['Repeat after me *'] , true).then((i,wildcard) => {
//     artyom.say("You've said : " + wildcard);
// });

// // or add some commandsDemostrations in the normal way
// artyom.addCommands([
//     {
//         indexes: ['Hello','Hi','is someone there'],
//         action: (i) => {
//             artyom.say("Hello, it's me");
//         }
//     },
//     {
//         indexes: ['Repeat after me *'],
//         smart:true,
//         action: (i,wildcard) => {
//             artyom.say("You've said : "+ wildcard);
//         }
//     },
//     // The smart commands support regular expressions
//     {
//         indexes: [/Good Morning/i],
//         smart:true,
//         action: (i,wildcard) => {
//             artyom.say("You've said : "+ wildcard);
//         }
//     },
//     {
//         indexes: ['shut down yourself'],
//         action: (i,wildcard) => {
//             artyom.fatality().then(() => {
//                 console.log("Artyom succesfully stopped");
//             });
//         }
//     },
// ]);

// // Start the commands !
// artyom.initialize({
//     lang: "en-GB", // GreatBritain english
//     continuous: true, // Listen forever
//     soundex: true,// Use the soundex algorithm to increase accuracy
//     debug: true, // Show messages in the console
//     executionKeyword: "and do it now",
//     listen: true, // Start to listen commands !

//     // If providen, you can only trigger a command if you say its name
//     // e.g to trigger Good Morning, you need to say "Jarvis Good Morning"
//     name: "Jarvis" 
// }).then(() => {
//     console.log("Artyom has been succesfully initialized");
// }).catch((err) => {
//     console.error("Artyom couldn't be initialized: ", err);
// });

/**
 * To speech text
//  */
// artyom.say("Hello, this is a demo text. The next text will be spoken in Spanish",{
//     onStart: () => {
//         console.log("Reading ...");
//     },
//     onEnd: () => {
//         console.log("No more text to talk");

//         // Force the language of a single speechSynthesis
//         artyom.say("Hola, esto está en Español", {
//             lang:"es-ES"
//         });
//     }
// });

// import * as tf from '@tensorflow/tfjs';
// import {Webcam} from './webcam';

// const faceapi = require("face-api.js");
// var $ = require("jquery");


// const webcam = new Webcam(document.getElementById('webcam'));


// let age_model;
// let video_element;

// let minConfidence = 0.6;
// let modelLoaded = false;


// async function init() {
//     try {
//         age_model = await tf.loadModel('./age_models/model.json')

//         faceapi.loadFaceDetectionModel('./models')
//         faceapi.loadFaceLandmarkModel('./models')

//         await webcam.setup()


//         video_element = $('#webcam').get(0)

//         console.log('initialization completed')
//         run()
//     } catch (e) {
//         console.log(e)
//     }
// }


// async function run() {
//     while (true) {
//         const {width, height} = faceapi.getMediaDimensions(video_element)
//         const input = await faceapi.toNetInput(video_element)

//         const canvas = $('#overlay').get(0)
//         canvas.width = width
//         canvas.height = height

//         const locations = await faceapi.locateFaces(input, minConfidence)
//         var faceImages = await faceapi.extractFaces(input.inputs[0], locations)

//         // detect landmarks and get the aligned face image bounding boxes
//         const alignedFaceBoxes = await Promise.all(faceImages.map(
//             async (faceCanvas, i) => {
//                 const faceLandmarks = await faceapi.detectLandmarks(faceCanvas)
//                 return faceLandmarks.align(locations[i])
//             }
//         ))
//         faceImages = await faceapi.extractFaces(input.inputs[0], alignedFaceBoxes)

//         // do things to each face
//         const gender_tag =  $('#gender')
//         const age_tag = $('#age')
//         faceImages.forEach(async (faceCanvas, i) => {
//                 const faceEl = $('#face')
//                 const ctx = canvas.getContext('2d')
//                 ctx.drawImage(faceCanvas, 0, 0)

//                 var img = tf.fromPixels(faceCanvas)
//                 const size = Math.min(img.shape[0], img.shape[1]);
//                 const centerHeight = img.shape[0] / 2;
//                 const beginHeight = centerHeight - (size / 2);
//                 const centerWidth = img.shape[1] / 2;
//                 const beginWidth = centerWidth - (size / 2);
//                 img = img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
//                 img = img.resizeBilinear([64, 64])
//                 img = img.expandDims(0);
//                 img = img.toFloat();
//                 //img = img.toFloat().div(tf.scalar(127)).sub(tf.scalar(1)); //Do not use this line 

//                 const results = age_model.predict(img)

//                 const predicted_genders = results[0].dataSync()
//                 if (predicted_genders[0] > 0.5) {
//                     gender_tag.val("Female")
//                     //console.log("Female")
//                 } else {
//                     gender_tag.val("Male")
//                     //console.log("Male")
//                 }
                
//                 const ages = tf.range(0, 101, 1).reshape([101, 1])
//                 const predicted_ages = results[1].dot(ages).flatten().dataSync()
//                 age_tag.val(predicted_ages[0] )
//                 //console.log("How old: ", predicted_ages[0] - 10)
//         })
        

//         faceapi.drawDetection('overlay', locations.map(det => det.forSize(width, height)))

//         document.getElementById('loading').style.display = 'none';
//         await tf.nextFrame()
//     }
// }




// //init()
