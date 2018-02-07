public class HiddenLayer extends NeuralLayer {

  /**
   * layer constructor with default sigmoid activation function
   * @param inputCount  number of inputs each neuron has to handle
   * @param neuronCount number of neurons in the layer
   */
  public HiddenLayer(int inputCount, int neuronCount) {
    super(inputCount, neuronCount);
  }

  /**
   * processes the layer input and returns the output of neurons as an array
   * @param in input data array
   * @return output of layer
   */
  @Override public double[] propagate(double[] in) {
    //feed the input through the neurons one at a time (room for improvement!)
    double[] ret = new double[getLayerNeuronCount()];
    for (int i = 0; i < ret.length; i++) {
      ret[i] = getNeuron(i).fire(in);
    }
    return ret;
  }

  /**
   * Object string representation
   * @return returns a string with info about the network size and type
   */
  @Override public String toString() {
    return super.toString() + " type: hidden";
  }
}
