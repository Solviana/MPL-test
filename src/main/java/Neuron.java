import java.util.Arrays;
import java.util.Random;

public class Neuron {

  private double[] weights;
  private double bias;
  private int inputCount;
  /*
   * question: is it a member of the neuron or the layer?
   * there might be memory issues
   */

  /**
   * creates a neuron with randomly initialized weights
   * @param inputCount number of inputs the neuron accepts
   */
  public Neuron(int inputCount) {
    this.inputCount = inputCount;
    this.weights = new double[inputCount];
    Random rand = new Random();

    //setting random weight values
    for (int i = 0; i < weights.length; i++) {
      weights[i] = (rand.nextDouble() - 0.5) * 0.1;
    }
    bias = (rand.nextDouble() - 0.5) * 0.1;
  }

  /**
   * getter for the number of inputs
   * @return number of inputs in neuron
   */
  public int getInputCount() {
    return this.inputCount;
  }

  /**
   * For testing purposes
   * @return string stating the number of inputs
   */
  @Override public String toString() {
    return "Number of inputs: " + inputCount;
  }

  /**
   * method to fire the neuron.
   * @param in neuron input array: has to be the same length as the weight array
   * @return neuron output: dot product of weight and input vectors
   */
  public double fire(double[] in) {
    if (in.length != this.weights.length) {
      throw new IllegalArgumentException("Inappropriate input length: Neuron");
    }

    double sum = 0;

    /* dot product of the weight and input arrays
    the loop multiplies the elements with same indices and adds them to the accumulator */
    for (int i = 0; i < in.length; i++) {
      sum += in[i] * this.weights[i];
    }
    sum += bias;

    return sum;
  }

  /**
   * method to fire the neuron with a single input. intended for input layer use
   * @param in neuron input parameter
   * @return weight and input vector dot product
   */
  public double fire(double in) {
    if (1 != this.weights.length) {
      throw new IllegalArgumentException("Inappropriate input length: Neuron");
    }
    return this.weights[0] * in + bias;
  }

  public double getWeight(int index) {
    if (index > weights.length) {
      throw new IndexOutOfBoundsException("Weight index problem: getWeight");
    }

    if (index == weights.length) {
      return bias;
    }

    else return weights[index];
  }

  protected void updateWeights(double[] delta) {
    if (delta.length != weights.length + 1) {
      throw new IllegalArgumentException("Neuron weight update error: inappropriate input");
    }
    for (int i = 0; i < this.weights.length; i++) {
      this.weights[i] += delta[i];
    }
    bias += delta[delta.length - 1];
  }

  public void setWeights(double[] weights) {
    if (weights.length != this.weights.length + 1) {
      throw new IllegalArgumentException("......");
    }
    this.weights = Arrays.copyOfRange(weights, 0, this.weights.length);
    this.bias = weights[weights.length-1];
  }
}