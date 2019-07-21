const xs = [];
const ys = [];
let coefficients = [];
const degree = 4;

const learningRate = 0.2;
const optimizer = tf.train.adam(learningRate);

function setup() {
  createCanvas(400, 400);

  for (let i = 0; i <= degree; i++) {
    coefficients[i] = tf.variable(tf.scalar(random(-1, 1)), true);
  }
}

function draw() {
  tf.tidy(() => {
    if (xs.length > 0) {
      const yValsTensor = tf.tensor1d(ys);
      optimizer.minimize(() => loss(predict(xs), yValsTensor));
    }
  });

  background(0);

  stroke(255);
  strokeWeight(4);

  for (let i = 0; i < xs.length; i++) {
    const x = map(xs[i], -1, 1, 0, width);
    const y = map(ys[i], -1, 1, height, 0);
    point(x, y);
  }

  const xpts = []
  for (let x = -1; x < 1; x += 0.05) {
    xpts.push(x);
  }
  
  const ypts = tf.tidy(() => predict(xpts));
  const yNums = ypts.dataSync();
  ypts.dispose();
  
  noFill();
  beginShape();  
  for (let i = 0; i < xpts.length; i++) {
    let x = map(xpts[i], -1, 1, 0, width);
    let y = map(yNums[i], -1, 1, height, 0);
    vertex(x, y);
  }
  endShape();
}

function loss(preds, labels) {
  return preds.sub(labels).square().mean();
}

function predict(xVals) {
  const tenseXs = tf.tensor1d(xVals);
  
  // y = ax^2 + bx + c // degree 2
  // y = ax^3 + bx^2 + cx + d //degree 3
  let yVals = tf.variable(tf.zerosLike(tenseXs));
  for (let i = 0; i < degree; i++) {
    const coef = coefficients[i];
    const pow_ts = tf.fill(tenseXs.shape, degree - i);
    const sum = tf.add(yVals, coefficients[i].mul(tenseXs.pow(pow_ts)));
    yVals.dispose();
    yVals = sum.clone();
  }
  return yVals;
}

function mousePressed() {
  xs.push(map(mouseX, 0, width, -1, 1));
  ys.push(map(mouseY, 0, height, 1, -1));
}