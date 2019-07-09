class Perceptron {
  // wl --> weights-length
  // lr --> learning rate
  constructor(wl, lr) {
    this.weights = new Array(wl);

    // init weights
    for (var i = 0; i < this.weights.length; i++) {
      this.weights[i] = random(-1, 1);
    }

    // set learning rate
    this.lr = lr;
  }

  guess(inputs) {
    // create sum of all values
    var sum = 0;
    for (var i = 0; i < inputs.length; i++) {
      sum += inputs[i] * this.weights[i];
    }

    // call activation-function
    return this.activate(sum);
  }

  activate(sum) {
    // return sign of sum -1 or 1
    if (sum > 0) return 1;
    else return -1;
  }

  train(inputs, desired) {
    // get the guessed result
    var guess = this.guess(inputs);

    // compute the error
    var error = desired - guess;

    // adjust weights based on error
    for (var i = 0; i < this.weights.length; i++) {
      this.weights[i] += this.lr * error * inputs[i];
    }
  }

  getWeights() {
    return this.weights;
  }
}
