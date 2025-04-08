import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class Example4 {

    // choosing the distribution law from the STSP family basing on inverse CDF

    private final double theta_0, p_0;
    private final int n, nTrials;
    private final double[] THETA, P;

    public Example4(double theta_0, double p_0, int n, int nSim) {

        this.theta_0 = theta_0;
        this.p_0 = p_0;
        this.n = n;
        this.nTrials = nSim;

        this.THETA = new double[nSim];
        this.P = new double[nSim];
    }

    private double FInv(double y, double theta, double p) {

        if (y < theta) {
            return Math.pow(y * Math.pow(theta, p - 1.), 1. / p);
        } else {
            return 1. - Math.pow((1. - y) * Math.pow(1. - theta, p - 1.), 1. / p);
        }
    }

    private double[] getTSPSample() {

        double[] s = new double[n];
        for (int i = 0; i < n; ++i) { s[i] = FInv(ThreadLocalRandom.current().nextDouble(), theta_0, p_0); }
        return s;
    }

    double d(double[] sortedSample, double theta, double p) {

        double[] inv = new double[n];
        for (int i = 0; i < n; ++i) { inv[i] = FInv((i + 0.5) / n, theta, p); }

        double res = 0.;
        for (int i = 0; i < n; ++i) {
            res += Math.pow(inv[i] - sortedSample[i], 2);
        }
        return res;
    }

    void trial(int trialIndex) {

        double thetaMin = 0., thetaMax = 1.0001, pMin = 0.5, pMax = 5.0001;

        double[] sample = getTSPSample();
        Arrays.sort(sample);

        double d0 = Double.MAX_VALUE;
        double theta0 = Float.NaN;
        double p0 = Float.NaN;

        for (double theta = thetaMin; theta < thetaMax; theta += 0.1) {
            for (double p = pMin; p < pMax; p += 0.5) {
                double d = d(sample, theta, p);
                if (d < d0) {
                    theta0 = theta;
                    p0 = p;
                    d0 = d;
                }
            }
        }

        // refining the estimate...
        thetaMin = theta0 - 0.1;
        thetaMax = Math.min(theta0 + 0.1001, 1.0001);
        pMin = Math.max(p0 - 0.5, 1.);
        pMax = p0 + 0.501;

        for (double theta = thetaMin; theta < thetaMax; theta += 0.01) {
            for (double p = pMin; p < pMax; p += 0.05) {
                double d = d(sample, theta, p);
                if (d < d0) {
                    theta0 = theta;
                    p0 = p;
                    d0 = d;
                }
            }
        }

        P[trialIndex] = p0;
        THETA[trialIndex] = theta0;
    }

    private static double mean(double[] x) {

        double s = 0.;
        for (double v: x) { s += v; }
        return s / x.length;
    }

    private static double stdev(double[] x) {

        double m = mean(x), s = 0.;
        for (double v: x) { s += Math.pow(v - m, 2); }
        return Math.sqrt(s / (x.length - 1));
    }

    private void runTrialsInParallel() {

        IntStream.range(0, nTrials).parallel().forEach(this::trial);
    }

    private void postProcessing() {
        System.out.printf("wMean: mean = %.3f, stdev = %.3f\n", mean(THETA), stdev(THETA));
        System.out.printf("MLE:   mean = %.3f, stdev = %.3f\n", mean(P), stdev(P));
    }


    public static void main(String[] args) {

        long t0 = System.currentTimeMillis();

        double theta_0 = 0.5, p_0 = 3.;
        int n = 50;

        Example4 test = new Example4(theta_0, p_0, n, 1_000_000);
        test.runTrialsInParallel();
        System.out.printf("done in %.2f s\n", 1.e-3 * (System.currentTimeMillis() - t0));
        test.postProcessing();
    }
}
