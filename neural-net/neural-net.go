// neuralNet contains all the information that defines a trained neural network
type neuralNet struct {
  config neuralNetConfig
  wHidden *mat.Dense
  bHidden *mat.Dense
  wOut *mat.Dense
  bOut *mat.Dense
}

// neuralNetConfig defines our neural network architecture and learning
// parameters
type neuralNetConfig struct {
  inputNeurons int
  outputNeurons int
  hiddenNeurons int
  numEpochs int
  learningRate float64
}

// newNetwork initializes a new neural network
func newNetwork(config neuralNetConfig) *neuralNet {
  return &neuralNet{config: config}
}

// sigmoid implements the sigmoid function for use in activation functions
func sigmoid(x float64) float64 {
  return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime implements the derivative of the sigmoid function for
// backpropogation
func sigmoidPrime(x float64) float64 {
  return x*(1.0-x)
}

// train trains a neural network using backpropogation
func (nn *neuralNet) train(x, y *mat.Dense) error {

  // Initialize biases/weights
  randSource := rand.NewSource(time.Now().UnixNano())
  randGen := rand.New(randSource)

  wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
  bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
  wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
  bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

  wHiddenRaw := wHidden.RawMatrix().Data

}
