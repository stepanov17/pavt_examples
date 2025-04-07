import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

// evaluating results of replicate measurements:
// compare weighted mean (The CIPM Mutual Recognition Arrangement (CIPM MRA), 1999;
// assumes the normal distributions for the measurand)
// vs maximum likelihood estimate (MLE)

public class Example2 {

    private final int nTrials;
    private final double[] MLE;  // MLE estimates
    private final double[] WMEAN;  // weighted mean

    private final double[] powers;

    private final int n;

    private final double[] r;  // TSP r parameter values
    private final double[] w;  // measurement weights
    private final double kW;   // to be used for weighted mean estimates

    public Example2(double[] tspPowers, double[] uncertainties, int nTrials) {

        this.nTrials = nTrials;
        this.WMEAN = new double[nTrials];
        this.MLE = new double[nTrials];

        this.powers = tspPowers;
        this.n = powers.length;

        this.r = new double[n];
        this.w = new double[n];
        double s = 0.;
        for (int i = 0; i < n; ++i)
        {
            double p = powers[i];
            r[i] = uncertainties[i] * Math.sqrt(0.5 * (p + 1.) * (p + 2.));
            w[i] = 1. / (uncertainties[i] * uncertainties[i]);
            s += w[i];
        }
        this.kW = 1. / s;

    }

    private final static double maxX0 = 0.2, dX0 = 2.e-5;
    private final static double minX0 = -maxX0;

    private static double getTSPValue(double r, double p) {
        // Generate random value from a symmetric TSP(-r, 0, r, p) (having a zero expectation)
        double u = ThreadLocalRandom.current().nextDouble();
        if (u < 0.5) { return r * (Math.pow(2. * u, 1. / p) - 1.); }
        else { return r * (1. - Math.pow(2. * (1. - u), 1. / p)); }
    }

    private static double pdf(double x, double x0, double r, double p) {

        double a = x0 - r, m = x0, b = x0 + r, k = 0.5 * p / r;
        if ((x > a) && (x <= m)) {
            return k * Math.pow((x - a) / (m - a), p - 1.);
        } else if ((x > m) && (x <= b)) {
            return k * Math.pow((b - x) / (b - m), p - 1.);
        }
        return 0.;
    }

    private double[] nextSample() {

        double[] res = new double[n];
        for (int i = 0; i < n; ++i) {
            res[i] = getTSPValue(r[i], powers[i]);
        }
        return res;
    }

    private double mle(double[] sample) {

        double eps = 1.e-10;

        double L0 = 0.;
        double m0 = -1.;

        int nM = (int)((maxX0 - minX0) / dX0) + 1;
        for (int im = 0; im < nM; ++im) {

            double m = minX0 + dX0 * im;

            double L = 1.;
            for (int i = 0; i < n; ++i) {
                double f = pdf(sample[i], m, r[i], powers[i]);
                if (Math.abs(f) < eps) {
                    L = 0.;
                    break;
                }
                L = L * f;
            }
            if ((L > eps) && (L > L0)) {
                L0 = L;
                m0 = m;
            }
        }

        return m0;
    }

    private double wMean(double[] sample) {

        double res = 0.;
        for (int i = 0; i < n; ++i) {
            res += sample[i] * w[i];
        }
        return kW * res;
    }

    private void trial(int trialIndex) {

        double[] x = nextSample();
        MLE[trialIndex] = mle(x);
        WMEAN[trialIndex] = wMean(x);

        if ((trialIndex + 1) % 1000 == 0) {
            System.out.println(">> " + (trialIndex + 1));
        }
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
        System.out.printf("wMean: mean = %.3f, stdev = %.3f\n", mean(WMEAN), stdev(WMEAN));
        System.out.printf("MLE:   mean = %.3f, stdev = %.3f\n", mean(MLE), stdev(MLE));
    }


    public static void main(String[] args) {

        double[] powers = {2.5, 2.6, 1.7, 3.8, 2.1};
        double[] uncertainties = {0.05, 0.07, 0.10, 0.15, 0.15};

        Example2 test = new Example2(powers, uncertainties, 1_000_000);

        long t0 = System.currentTimeMillis();
        test.runTrialsInParallel();
        test.postProcessing();
        System.out.printf("done in %.2f s\n", 1.e-3 * (System.currentTimeMillis() - t0));

        // save MLE, WMEAN to files, plot histograms etc.
    }
}
