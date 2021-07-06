'use strict';

var robot = require("robotjs");

var tf = require('@tensorflow/tfjs');
const screenshot = require('screenshot-desktop')
const cocoSsd = require('@tensorflow-models/coco-ssd');
// Build and compile model.
const model = tf.sequential();

var imgLen = 439491;
model.add(tf.layers.dense({units: 2, inputShape: [3, 1080, 1920], activation: "relu", dropout: 0.00001 }));
model.add(tf.layers.conv2d({
    filters: 256,
    kernelSize: 3,
    strides: 1,
    activation: 'relu'
  }));
  
model.add(tf.layers.flatten({shape: [3,1080,1920]}));
  model.add(tf.layers.dense({units: 100, activation: 'relu'}));
  model.add(tf.layers.dropout({rate: 0.25}));
model.add(tf.layers.dense({units: 2, activation: "relu", dropout: 0.00001 }));

model.compile({optimizer: tf.train.rmsprop(0.00001), loss: tf.losses.meanSquaredError });
let c;
(async () => {
c = await cocoSsd.load();
})()

let img;
var canTrain = true;
let xs;
let m = {x:0, y:0}
function shot() {
screenshot({format: 'png'}).then(async (img) => {
    if (!canTrain) return;
    var png = tf.node.decodePng(img).reshape([1, 3, 1080, 1920])

    
    xs = png;
    var mouse = robot.getMousePos(); 
    var d = { x: m.x - mouse.x, y: m.y - mouse.y }
    m = mouse
    var ys = tf.tensor2d([d.x, d.y], [1, 2]);
   // model.fit(xs, ys, { epochs: 2, batchSize: 1 } ).then(() => canTrain = true )
   // canTrain = false;

})
}

setInterval(shot, 25);
