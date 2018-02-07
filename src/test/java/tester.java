public class tester {
  public static void main(String... args) {
    NeuralNetwork network = new NeuralNetwork(8, 8);
    network.addHiddenLayer(3);

    double[][] trainingData = {
        {1,0,0,0,0,0,0,0, 0,1,0,0,0,0,0,0},
        {0,1,0,0,0,0,0,0, 0,0,1,0,0,0,0,0},
        {0,0,1,0,0,0,0,0, 0,0,0,1,0,0,0,0},
        {0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0},
        {0,0,0,0,1,0,0,0, 0,0,0,0,0,1,0,0},
        {0,0,0,0,0,1,0,0, 0,0,0,0,0,0,1,0},
        {0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1},
        {0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0}
    };

    double[] test = {1,0,0,0,0,0,0,0};
    double[] out;
    long l = System.currentTimeMillis();
    network.train(trainingData,trainingData,0.05, 60000);
    System.out.println(System.currentTimeMillis() - l);
    out = network.classify(test);
    for (double d : out) {
      System.out.print(d + " ");
    }
  }
}
