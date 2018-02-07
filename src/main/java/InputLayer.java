
public class InputLayer extends NeuralLayer {

  /**
   * constructor that defaults to linear activation function
   * the neurons in this layer simply pass the given data input (1 per neuron) to the first hidden
   * layer of neurons
   * @param inputCount number of neural network inputs: every neuron handles a single input
   */
  public InputLayer(int inputCount) {
    super(1, inputCount, x->x);

    //fix weight to 1 bias to 0
    for (int i = 0; i < inputCount; i++) {
      double[] weight = {1, 0};
      this.getNeuron(i).setWeights(weight);
    }
  }

  /**
   * processes the layer input and returns the output of neurons as a
   * @param in input data array
   * @return output of layer
   */
  @Override public double[] propagate(double[] in) {
    //error checking
    if (in.length != getLayerNeuronCount()) {
      throw new IllegalArgumentException("Inappropriate array length: InputLayer");
    }

    //feed the input through the neurons one at a time (room for improvement!)
    double[] ret = new double[getLayerNeuronCount()];
    for (int i = 0; i < ret.length; i++) {
      ret[i] = getNeuron(i).fire(in[i]);
    }

    return ret;
  }

  /**
   * Object string representation
   * @return returns a string with info about the network size and type
   */
  @Override public String toString() {
    return super.toString() + " type: input";
  }
}
