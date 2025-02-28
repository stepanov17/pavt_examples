import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

// obtain the bias of the maximum likelihood estimates of the parameters of the STSP(theta, p) distribution
// (STSP: S. Kotz, J.R. Van Dorp. Beyond Beta: Other Continuous Families of Distributions with
// Bounded Support and Applications. World Scientific Publishing, 2004)


public class Example3 {

    private final double theta0, p0;
    private final int n, nTrials;

    private final double[] THETA;
    private final double[] P;

    public Example3(double theta, double p, int n, int nSim) {

        this.theta0 = theta;
        this.p0 = p;
        this.n = n;
        this.nTrials = nSim;

        this.THETA = new double[nSim];
        this.P = new double[nSim];
    }

    private double[] getSTSPSample() {

        double[] res = new double[n];
        for (int i = 0; i < n; ++i) {
            double y = ThreadLocalRandom.current().nextDouble();
            if (y < theta0) {
                res[i] = Math.pow(y * Math.pow(theta0, p0 - 1.), 1. / p0);
            } else {
                res[i] = 1. - Math.pow((1. - y) * Math.pow(1. - theta0, p0 - 1.), 1. / p0);
            }
        }
        return res;
    }

    private int argMax(double[] x) {

        int i0 = 0;
        double max = x[0];

        for (int i = 1; i < x.length; ++i) {
            if (x[i] > max) {
                i0 = i;
                max = x[i];
            }
        }
        return i0;
    }

    private void MLEst(int i) {

        double[] s = getSTSPSample();
        Arrays.sort(s);

        double[] M = new double[n];
        for (int r = 0; r < n; ++r) {
            double s_r = s[r];
            double m = 1.;
            for (int j = 0; j < r; ++j) {
                m *= (s[j] / s_r);
            }
            for (int j = r + 1; j < n; ++j) {
                m *= ((1. - s[j]) / (1. - s_r));
            }
            M[r] = m;
        }
        int i0 = argMax(M);
        THETA[i] = s[i0];
        P[i] = -n / Math.log(M[i0]);
   }

    private double mean(double[] x) {

        double s = 0.;
        for (double v: x) { s += v; }
        return s / x.length;
    }

    private double stdev(double[] x) {

        int nx = x.length;
        double m = mean(x);
        double s = 0.;
        for (double v: x) { s += Math.pow(v - m, 2); }
        return Math.sqrt(s / (nx - 1));
    }


    private void runTrialsInParallel() {

        IntStream.range(0, nTrials).parallel().forEach(this::MLEst);

        // post-processing: calculate mean(bias), stdev(bias) and correction factor

        double pBias = mean(P) - p0;
        double pBiasPerc = 100. * pBias / p0;

        double thetaBias = mean(THETA) - theta0;
        double thetaBiasPerc = 100. * thetaBias / theta0;

        System.out.printf("p0 = %.1f:     \t bias = %.3f (%.1f%%) \t stdev = %.3f \t k = %.2f\n",
                p0, pBias, pBiasPerc, stdev(P), p0 / mean(P));

        System.out.printf("theta0 = %.1f: \t bias = %.3f (%.1f%%) \t stdev = %.3f \t k = %.2f\n\n",
                theta0, thetaBias, thetaBiasPerc, stdev(THETA), theta0 / mean(THETA));
    }


    public static void main(String[] args) {

        long t0 = System.currentTimeMillis();

        int n = 50;
        double[] P = {2., 3., 5.};

        for (double p: P) {
            for (double theta = 0.1; theta < 0.501; theta += 0.1) {
                new Example3(theta, p, n, 3_000_000).runTrialsInParallel();
            }
        }

        System.out.printf("done in %.2f s\n", 1.e-3 * (System.currentTimeMillis() - t0));
    }
}
