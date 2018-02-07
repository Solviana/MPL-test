import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleFunction;

public class Neuron {

  private double[] weights;
  private double bias;
  private int inputCount;
  /*
   * question: is it a member of the neuron or the layer?
   * there might be memory issues
   */
  private DoubleFunction<Double>  activationFunction;

  /**
   * creates a neuron with randomly initialized (between -0.5 and 0.5) weights
   * @param inputCount number of inputs the neuron accepts
   * @param activationFunction neuron's activation function
   */
  public Neuron(int inputCount, DoubleFunction activationFunction) {
    this.inputCount = inputCount;
    this.activationFunction = activationFunction;
    this.weights = new double[inputCount];
    Random rand = new Random();

    //setting random weight values
    for (int i = 0; i < weights.length; i++) {
      weights[i] = rand.nextDouble() - 0.5;
    }
    bias = rand.nextDouble() - 0.5;
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
   * @return neuron output: dot product of weight and input vectors, fed to the activation function
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

    double ret = this.activationFunction.apply(sum);
    return ret;
  }

  /**
   * method to fire the neuron with a single input. intended for input layer use
   * @param in neuron input parameter
   * @return
   */
  public double fire(double in) {
    if (1 != this.weights.length) {
      throw new IllegalArgumentException("Inappropriate input length: Neuron");
    }
    return this.activationFunction.apply(this.weights[0] * in + bias);
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