import java.util.function.DoubleFunction;

// to be implemented...
public class ActivationFunction {
  private DoubleFunction<Double> function;
  private DoubleFunction<Double> derivative;

  private ActivationFunction(DoubleFunction<Double> function, DoubleFunction<Double> derivative) {
    this.function = function;
    this.derivative = derivative;
  }

  public double apply(double x) {
    return function.apply(x);
  }

  public double applyDerivative(double x) {
    return derivative.apply(x);
  }
}
