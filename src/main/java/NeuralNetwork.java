import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * framework for MLP-s (Multiple Layer Perceptron)
 */
public class NeuralNetwork {

  private int inputCount;
  private int outputCount;
  private ArrayList<NeuralLayer> layers;

  /**
   * network constructor.
   * default network size is 2 (input and output layers), additional hidden layers can be added
   * with methods
   * @param inputCount number of inputs in the network
   * @param outputCount number of outputs in the network
   */
  public NeuralNetwork(int inputCount, int outputCount) {
    layers = new ArrayList<>();
    layers.add(new InputLayer(inputCount));
    layers.add(new HiddenLayer(inputCount, outputCount));
    this.inputCount = inputCount;
    this.outputCount = outputCount;
  }

  /**
   * getter for network size
   * @return number of layers in the network
   */
  public int getLayerCount() {
    return layers.size();
  }

  /**
   * getter for network layers
   * @param index index of layer to be returned
   * @return layer at index
   */
  public NeuralLayer getLayer(int index) {
    if (index >= layers.size()) {
      throw new IndexOutOfBoundsException();
    }
    return this.layers.get(index);
  }

  /**
   * returns the index of the last element.
   * this method is only here to improve the readability so its private
   * @return index of the output layer
   */
  private int getLastIndex() {
    return getLayerCount() - 1;
  }

  /**
   * adds a layer before the output layer, and updates the output layer to have appropriate
   * amount of inputs
   * @param l layer to be added
   *          layer neurons have to have appropriate inputs
   */
  public void addHiddenLayer(HiddenLayer l) {
    if (l.getNeuron(0).getInputCount() != layers.get(getLastIndex() - 1).getLayerNeuronCount()) {
      throw new IllegalArgumentException("Inappropriate input count");
    }
    
    layers.add(getLastIndex(), l);
    HiddenLayer newOutLayer = new HiddenLayer(l.getLayerNeuronCount(),
        this.outputCount);
    layers.set(getLastIndex(), newOutLayer);
  }

  /**
   * adds a hidden layer with n neurons
   * @param neuronCount number of neurons in the layer to be added
   */
  public void addHiddenLayer(int neuronCount) {
    int layerInputCount = layers.get(getLastIndex() - 1).getLayerNeuronCount();
    HiddenLayer layer = new HiddenLayer(layerInputCount, neuronCount);
    addHiddenLayer(layer);
  }

  /**
   * feeds data through the network and returns the output
   * @param in input data
   *           has to have as many elements as there are inputs in the network
   * @return returns the network output
   */
  public double[] classify(double[] in){
    if (in.length != inputCount) {
      throw new IllegalArgumentException("Cant classify: input number mismatch");
    }

    // creating return array and feeding it through the layers
    double[] ret = in;
    for (NeuralLayer n : layers) {
      ret = n.propagate(ret);
    }
    return ret;
  }

  /**
   * training algorithm for the MLP, implementing backpropogation. See any machine learning book
   * for details ('Tom M. Mitchell - Machine learning' is decent)
   * @param trainingData data to train the network on. format:
   *                     rows are independent training examples
   *                     columns are inputs/outputs, first the inputs, then the outputs
   * @param validationData data to validate the network. same form as the training data
   * @param trainingRate see any ML book
   */
  public void train(final double[][] trainingData, final double[][] validationData, double
      trainingRate, int maxIterations){
    // only checking the first data pair, might have to put this in a loop to check each input
    if (trainingData[0].length != inputCount + outputCount || trainingRate <= 0 ||
        trainingData[0].length != inputCount + outputCount) {
      throw new IllegalArgumentException("Invalid training data");
    }
    String filePath = "error.txt";
    try(final FileWriter outFile = new FileWriter(filePath)) {
      double previousError;
      double error = trainingValidation(validationData);
      outFile.write(error + System.lineSeparator());
      int i = 0;
      do {
        previousError = error;
        trainOnBatch(trainingData, trainingRate);
        error = trainingValidation(validationData);
        outFile.write(error + System.lineSeparator());
        i++;
      } while (previousError > error && i < maxIterations);
    } catch (IOException e) {
      System.out.println("IOException");
    }
  }

  private void trainOnBatch(double[][] trainingData, double trainingRate) {
    // iterate over the examples one at a time, and update the weights using backpropagation
    for (double[] example : trainingData) {
      // stack to hold the layer results
      ArrayDeque<double[]> results;
      // array to hold the neuron output derivatives
      double[] neuronDelta;
      results = new ArrayDeque<>();
      // separate input from output
      double[] trainingInput = Arrays.copyOfRange(example, 0, inputCount);
      double[] trainingOutput = Arrays.copyOfRange(example, inputCount, example.length);
      results.push(trainingInput);
      // propagating data through the network, while pushing every layer's output onto the stack
      for (NeuralLayer n : layers) {
        results.push(n.propagate(results.peek()));
      }

      // process the output layer and update its weights
      // variable to hold the current layer reference
      NeuralLayer currentLayer = layers.get(layers.size() - 1);
      neuronDelta = new double[currentLayer.getLayerNeuronCount()];
      // 2D array to hold the weight updates
      double[][] weightDelta = new double[currentLayer.getLayerNeuronCount()][currentLayer
          .getNeuron(0).getInputCount() + 1];
      // current layer's output, popped from the stack
      double[] output = results.pop();
      for (int i =  0; i < currentLayer.getLayerNeuronCount(); i++) {
        // calculate the neuron's derivative, the output layer is linear so it's easy
        neuronDelta[i] = (trainingOutput[i] - output[i]) * output[i] * (1 - output[i]);
        for (int j = 0; j < currentLayer.getNeuron(i).getInputCount(); j++) {
          // calculate the gradient
          weightDelta[i][j] = neuronDelta[i] * trainingRate * results.peek()[j];
        }
        // bias delta
        weightDelta[i][currentLayer.getNeuron(i).getInputCount()] = neuronDelta[i] * trainingRate;
      }
      // if there are no hidden layers in the network, we can update the weights since they wont
      // be used in the next loop
      if(layers.size() == 2) {
        currentLayer.updateLayerWeights(weightDelta);
      }
      // repeat for hidden layers, if present (do not touch the input layer!)
      for (int i = layers.size() - 2; i > 0; i--) {
        // store the reference to the current and previous layer for future use
        currentLayer = layers.get(i);
        NeuralLayer previousLayer = layers.get(i + 1);
        // new variable for holding the weight updates. can't use the old one, because the
        // weights can only update when this layer is finished
        double[][] newWeightDelta = new double[currentLayer.getLayerNeuronCount()][currentLayer
            .getNeuron(0).getInputCount() + 1];
        // new variable for holding the deltas. can't use the old one because we need its values
        // to calculate the deltaSums
        double[] newNeuronDelta = new double[currentLayer.getLayerNeuronCount()];
        output = results.pop();
        // take each neuron and calculate its error (derivative) and weight updates
        for (int j =  0; j < currentLayer.getLayerNeuronCount(); j++) {
          double deltaSum = 0;
          // calculate the sum of output errors
          for (int k = 0; k < previousLayer.getLayerNeuronCount(); k++) {
            deltaSum += previousLayer.getNeuron(k).getWeight(j) * neuronDelta[k];
          }
          // calculate the neuron's derivative
          newNeuronDelta[j] = output[j] * (1 - output[j]) * deltaSum;
          for (int k = 0; k < currentLayer.getNeuron(j).getInputCount(); k++) {
            // calculate the gradient
            newWeightDelta[j][k] = newNeuronDelta[j] * trainingRate * results.peek()[k];
          }
          // bias delta
          newWeightDelta[j][currentLayer.getNeuron(j).getInputCount()] = newNeuronDelta[j] *
              trainingRate;
        }
        // the calculation is finished, we dont need the previous layer's old weights anymore, so
        // we can update them
        previousLayer.updateLayerWeights(weightDelta);
        // on the last iteration update current layer's weights
        if(i == 1) {
          currentLayer.updateLayerWeights(newWeightDelta);
        }
        // otherwise update the variables and move onto the next iteration
        else {
          // move the data into the old variables so we can use them in the next iteration
          weightDelta = newWeightDelta;
          neuronDelta = newNeuronDelta;
        }
      }
    }
  }

  /**
   * calculates the error function on a given set of validation data
   * @param validationData valid input-output data
   * @return sum of the error function (t-o)^2/2 over every output and example
   */
  private double trainingValidation(final double[][] validationData) {
    double ret = 0;
    double[][] output = new double[validationData.length][];
    // separate input from output
    for (int i = 0; i < validationData.length; i++) {
      double[] validationInput = Arrays.copyOfRange(validationData[i],0, inputCount);
      double[] validationOutput = Arrays.copyOfRange(validationData[i], inputCount,
          validationData[i].length);
      output[i] = this.classify(validationInput);
      for (int j  = 0; j < output[i].length; j++) {
        ret += Math.pow(output[i][j] - validationOutput[j], 2) / 2;
      }
    }
    return ret;
  }

  /**
   * Object string representation
   * @return returns a string with info about the network size
   */
  @Override public String toString() {
    String ret =  "Network has " + getLayerCount() + " layers";
    for (NeuralLayer n: this.layers) {
      ret += "\n" + n.toString();
    }
    return ret;
  }
}
