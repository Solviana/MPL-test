import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

/*
 * network tester with the mnist dataset.
 * takes
 * output: 10 values representing digits (0...9)
 * 0.1 means false 0.9 true (easier training on sigmoids, can be rounded to the nearest integer)
 * console output: results for every digit
 * network prediction:valid value
 */
public class tester2 {

  public static void main(String... args) {

    NeuralNetwork network = new NeuralNetwork(28 * 28, 10, ActivationFunction.sigmoid());
    network.addHiddenLayer(30);
    network.addHiddenLayer(20);

    try {
      double[][] trainingData = trainingDataLoader();
      double[][] validationData = validationDataLoader();

    long l = System.currentTimeMillis();
    network.train(trainingData, validationData, 0.01, 100);
    System.out.println("training time: " + (System.currentTimeMillis() - l));
    for (int i = 0; i < 10; i++) {
      test(network, validationData[i]);
    }
    } catch (IOException e) {
      System.out.println("Szar van a levesben");
    }
  }

  public static double[][] trainingDataLoader() throws IOException {
    // training set contains 60000 images
    double[][] trainingData = new double[6000][28 * 28 + 10];
    InputStream trainStream = tester2.class.getResourceAsStream("/train-images.idx3-ubyte");
    InputStream trainLabelStream = tester2.class.getResourceAsStream("/train-labels.idx1-ubyte");
    trainStream.skip(16);
    trainLabelStream.skip(8);

    // load training images
    for (int i = 0; i < trainingData.length; i++) {
      Arrays.fill(trainingData[i], 0.1);
      for (int j = 0; j < 28 * 28; j++) {
        trainingData[i][j] = (double) trainStream.read();
      }
      // set the expected output value to 0.9, rest stays 0;
      trainingData[i][28 * 28 + trainLabelStream.read()] = 0.9;
    }
    trainStream.close();
    trainLabelStream.close();

    return trainingData;
  }

  public static double[][] validationDataLoader() throws IOException {
    // test set contains 10000 images
    double[][] validationData = new double[1000][28 * 28 + 10];
    InputStream validStream = tester2.class.getResourceAsStream("/t10k-images.idx3-ubyte");
    InputStream validLabelStream = tester2.class.getResourceAsStream("/t10k-labels.idx1-ubyte");
    validStream.skip(16);
    validLabelStream.skip(8);

    // load validation images
    for (int i = 0; i < validationData.length; i++) {
      Arrays.fill(validationData[i], 0.1);
      for (int j = 0; j < 28 * 28; j++) {
        validationData[i][j] = (double) validStream.read();
      }
      // set the expected output value to 1, rest stays 0;
      validationData[i][28 * 28+ validLabelStream.read()] = 0.9;
    }
    validStream.close();
    validLabelStream.close();

    return validationData;
  }

  public static void test(NeuralNetwork network, double[] testData) {
    // seperate image from label
    double[] image = Arrays.copyOfRange(testData, 0, 28 * 28);
    double[] label = Arrays.copyOfRange(testData, 28 * 28, testData.length);
    double[] result = network.classify(image);
    // dump result and expected result onto console
    for(int i = 0; i < label.length; i++) {
      System.out.print(result[i] + ":" + label[i] + "\t");
    }
    System.out.println("\t|");
  }
}
