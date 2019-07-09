// Thomas Miller
// Simple Perceptron Example

// Credit: TheCodingTrain https://www.youtube.com/watch?v=ntKn5TPHHAk&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh&index=2

// list of points will be used for training
var trainingPts = new Array(2000);

// Perceptron obj
var perceptron;

// perceptron will be trained one point at a time
let count = 0;

// Coordinate space
let xmin = -1;
let ymin = -1;
let xmax = 1;
let ymax = 1;

// function to describe a line
function f(x) {
  let y = 0.3 * x + 0.4;
  return y;
}

function setup() {
  createCanvas(800, 800);

  // create Perceptron obj
  perceptron = new Perceptron(3, 0.1);

  // init random training points
  for (var i = 0; i < trainingPts.length; i++) {
    var x = random(xmin, xmax);
    var y = random(ymin, ymax);

    // calculate correct output
    var desired = 1;
    if (y < f(x)) desired = -1;

    // init pt in trainingPts
    trainingPts[i] = { input: [x, y, 1], output: desired };
  }
}

function draw() {
  // draw stuff ...
  background(0);

  // draw the line
  strokeWeight(1);
  stroke(255);
  var x1 = map(xmin, xmin, xmax, 0, width);
  var y1 = map(f(xmin), ymin, ymax, height, 0);
  var x2 = map(xmax, xmin, xmax, 0, width);
  var y2 = map(f(xmax), ymin, ymax, height, 0);

  line(x1, y1, x2, y2);

  // draw a line based on the current weights
  strokeWeight(2);
  stroke(255);

  var weights = perceptron.getWeights();

  x1 = xmin;
  y1 = (-weights[2] - weights[0] * x1) / weights[1];
  x2 = xmax;
  y2 = (-weights[2] - weights[0] * x2) / weights[1];

  x1 = map(x1, xmin, xmax, 0, width);
  y1 = map(y1, ymin, ymax, height, 0);
  x2 = map(x2, xmin, xmax, 0, width);
  y2 = map(y2, ymin, ymax, height, 0);

  line(x1, y1, x2, y2);

  // train Perceptron with one point at a time
  perceptron.train(trainingPts[count].input, trainingPts[count].output);
  count = (count + 1) % trainingPts.length;

  for (var i = 0; i < count; i++) {
    stroke(255);
    strokeWeight(1);
    fill(255);
    var guess = perceptron.guess(trainingPts[i].input);
    if (guess > 0) noFill();

    var x = map(trainingPts[i].input[0], xmin, xmax, 0, width);
    var y = map(trainingPts[i].input[1], ymin, ymax, height, 0);
    ellipse(x, y, 8, 8);
  }
}
