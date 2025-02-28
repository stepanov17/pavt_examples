import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

// calculate critical values for range(sample[k]) / stdev(sample[n]),
// where the samples are from the same symmetric TPS(r, p) distribution


public class Example6 {

    private final int n, k, nTrials;
    private final double p, r;
    private final double [] out;


    public Example6(double p, int n, int k, int nTrials) {

        this.p = p;
        this.r = 1.;  // the result does not depend on r value
        this.n = n;
        this.k = k;
        this.nTrials = nTrials;
        out = new double[nTrials];
    }

    private double[] getTSPSample(int len) {
        // generate a sample from TSP(r, p) distribution (assuming location parameter equal to 0)

        double[] res = new double[len];
        for (int i = 0; i < len; ++i) {
            double y = ThreadLocalRandom.current().nextDouble();
            // inverse CDF for symmetric TSP(r, p)
            if (y <= 0.5) {
                res[i] = r * (Math.pow(2. * y, 1. / p) - 1.);
            } else {
                res[i] = r * (1. - Math.pow(2. * (1. - y), 1. / p));
            }
        }
        return res;
    }

    private void trial(int i) {

        double s = stdev(getTSPSample(n));
        double r = range(getTSPSample(k));
        out[i] = r / s;
    }

    private double range(double[] x) {
        // sample range

        double min = x[0], max = x[0];
        for (double v : x) {
            if (v < min) { min = v; }
            else if (v > max) { max = v; }
        }
        return max - min;
    }

    private double mean(double[] x) {

        double s = 0.;
        for (double v: x) { s += v; }
        return s / x.length;
    }

    private double stdev(double[] x) {

        double m = mean(x);
        double s = 0.;
        for (double v: x) { s += Math.pow(v - m, 2); }
        return Math.sqrt(s / (x.length - 1));
    }

    private void runTrialsInParallel() {

        IntStream.range(0, nTrials).parallel().forEach(this::trial);

        // post-processing (get critical values for r/s ratio)

        Arrays.sort(out);
        int i1 = (int) Math.round(nTrials * 0.95);  // critical value for probability level of 95%
        int i2 = (int) Math.round(nTrials * 0.99);  // critical value for probability level of 95%
        System.out.printf("p = %.1f:  C(0.95) = %.2f  C(0.99) = %.2f\n", p, out[i1], out[i2]);
    }


    public static void main(String[] args) {

        long t0 = System.currentTimeMillis();

        final int k = 10, n = 50;
        double[] P = {0.5, 1., 1.5, 2., 2.5, 3., 5.};
        for (double p: P) {
            new Example6(p, n, k, 3_000_000).runTrialsInParallel();
        }

        System.out.printf("done in %.2f s\n", 1.e-3 * (System.currentTimeMillis() - t0));
    }
}
