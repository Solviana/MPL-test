import java.util.function.DoubleFunction;
import java.util.ArrayDeque;

public abstract class NeuralLayer {

  private Neuron[] neurons;

  /**
   * layer constructor
   * @param inputCount  number of inputs each neuron has to handle
   * @param neuronCount number of neurons in the layer
   * @param activationFunction activation function for the layer's neurons
   */
  protected NeuralLayer(int inputCount, int neuronCount, DoubleFunction<Double> activationFunction) {
    neurons = new Neuron[neuronCount];

    //initializing the neurons with random weigths
    for (int i = 0; i < neurons.length; i++) {
      neurons[i] = new Neuron(inputCount, activationFunction);
    }
  }

  /**
   * layer constructor with sigmoid activation function as default
   * can use this(..) constructor????
   * @param inputCount  number of inputs each neuron has to handle
   * @param neuronCount number of neurons in the layer
   */
  public NeuralLayer(int inputCount, int neuronCount) {
    this(inputCount, neuronCount, x -> 1 / (1 + Math.exp(-x)));
  }

  /**
   * getter for determining the number of outputs
   * @return number of neurons in the layer
   */
  public int getLayerNeuronCount() {
    return neurons.length;
  }

  /**
   * getter for individual neurons
   * @param index the index of the neuron returned
   * @return neuron at index
   */
  public Neuron getNeuron(int index) {
    return neurons[index];
  }


  /**
   * this function propagates data through the network
   * implementations may be different depending on layer type
   * @param in input data array
   * @return returns the output values of the neurons
   */
  public abstract double[] propagate(double[] in);

  /**
   * modifies the neuron weights by delta
   * @param delta array containing the weight modifications for each neuron (row) and each weight
   *             in the neuron (column)
   */
  protected void updateLayerWeights(double[][] delta) {
    if(delta.length != neurons.length) {
      throw new IllegalArgumentException("Layer weight update error: inappropriate input");
    }

    for(int i = 0; i < neurons.length; i++) {
      neurons[i].updateWeights(delta[i]);
    }
  }

  /**
   * Object string representation
   * @return returns a string with info about the layer size
   */
  @Override public String toString() {
    return "Number of neurons: " + neurons.length;
  }
}


