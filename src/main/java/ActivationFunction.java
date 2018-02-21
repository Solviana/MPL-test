
import java.util.function.DoubleFunction;

public class ActivationFunction {
  private DoubleFunction<Double> function;
  private DoubleFunction<Double> derivative;
  private String functionType;

  private ActivationFunction(DoubleFunction<Double> function, DoubleFunction<Double> derivative,
                             String functionType) {
    this.function = function;
    this.derivative = derivative;
    this.functionType = functionType;
  }

  public static ActivationFunction sigmoid() {
    return new ActivationFunction(x -> 1.0 / (Math.exp(-x) + 1),
        x -> Math.exp(-x) / Math.pow(Math.exp(-x) + 1.0, 2), "Sigmoid");
  }

  public static ActivationFunction linear() {
    return new ActivationFunction(x -> x,
        x -> 1.0, "Linear");
  }

  public static ActivationFunction rectifier() {
    return new ActivationFunction(x -> x > 0 ? x : 0,
        x -> x > 0 ? 1.0 : 0, "Rectifier");
  }

  public double apply(double x) {
    return function.apply(x);
  }

  public double applyDerivative(double x) {
    return derivative.apply(x);
  }
}
