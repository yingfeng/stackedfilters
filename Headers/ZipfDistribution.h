//
// Created by kylebd99 on 8/30/19.
//

#ifndef STACKEDALTERNATINGAMQS_ZIPFDISTRIBUTION_H
#define STACKEDALTERNATINGAMQS_ZIPFDISTRIBUTION_H

#include <cmath>

double r8_zeta(double p)

//****************************************************************************80
//
//  Purpose:
//
//    R8_ZETA estimates the Riemann Zeta function.
//
//  Discussion:
//
//    For 1 < P, the Riemann Zeta function is defined as:
//
//      ZETA ( P ) = Sum ( 1 <= N < oo ) 1 / N^P
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 October 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Daniel Zwillinger, editor,
//    CRC Standard Mathematical Tables and Formulae,
//    30th Edition,
//    CRC Press, 1996.
//
//  Parameters:
//
//    Input, double P, the power to which the integers are raised.
//    P must be greater than 1.  For integral P up to 20, a
//    precomputed value is returned; otherwise the infinite
//    sum is approximated.
//
//    Output, double R8_ZETA, an approximation to the Riemann
//    Zeta function.
//
{
    int n;
    const double r8_huge = 1.0E+30;
    const double r8_pi = 3.14159265358979323;
    double value;
    double zsum;
    double zsum_old;

    if (p <= 1.0) {
        value = r8_huge;
    } else if (p == 2.0) {
        value = pow(r8_pi, 2) / 6.0;
    } else if (p == 3.0) {
        value = 1.2020569032;
    } else if (p == 4.0) {
        value = pow(r8_pi, 4) / 90.0;
    } else if (p == 5.0) {
        value = 1.0369277551;
    } else if (p == 6.0) {
        value = pow(r8_pi, 6) / 945.0;
    } else if (p == 7.0) {
        value = 1.0083492774;
    } else if (p == 8.0) {
        value = pow(r8_pi, 8) / 9450.0;
    } else if (p == 9.0) {
        value = 1.0020083928;
    } else if (p == 10.0) {
        value = pow(r8_pi, 10) / 93555.0;
    } else if (p == 11.0) {
        value = 1.0004941886;
    } else if (p == 12.0) {
        value = 1.0002460866;
    } else if (p == 13.0) {
        value = 1.0001227133;
    } else if (p == 14.0) {
        value = 1.0000612482;
    } else if (p == 15.0) {
        value = 1.0000305882;
    } else if (p == 16.0) {
        value = 1.0000152823;
    } else if (p == 17.0) {
        value = 1.0000076372;
    } else if (p == 18.0) {
        value = 1.0000038173;
    } else if (p == 19.0) {
        value = 1.0000019082;
    } else if (p == 20.0) {
        value = 1.0000009540;
    } else {
        zsum = 0.0;
        n = 0;

        for (;;) {
            n = n + 1;
            zsum_old = zsum;
            zsum = zsum + 1.0 / pow((double) n, p);
            if (zsum <= zsum_old) {
                break;
            }
        }
        value = zsum;
    }

    return value;
}


double zeta_cdf(int x, double a)
//****************************************************************************80
//
//  Purpose:
//
//    ZIPF_CDF evaluates the Zipf CDF.
//
//  Discussion:
//
//    Simple summation is used.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 October 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int X, the argument of the PDF.
//    1 <= N
//
//    Input, double A, the parameter of the PDF.
//    1.0 < A.
//
//    Output, double CDF, the value of the CDF.
//
{
    double c;
    double cdf;
    double pdf;
    int y;

    if (x < 1) {
        cdf = 0.0;
    } else {
        c = r8_zeta(a);
        cdf = 0.0;
        for (y = 1; y <= x; y++) {
            pdf = 1.0 / pow(y, a) / c;
            cdf = cdf + pdf;
        }

    }

    return cdf;
}


long double zipf_denominator(size_t n, double a) {
    double denominator = 0;
    for (double i = 0; i < n; i++) {
        denominator += (double) 1 / std::pow(i, a);
    }
    return denominator;
}

long double zipf_numerator(size_t x, double a) {
    long double numerator = 0;
    for (double i = 0; i < x; i++) {
        numerator += (double) 1 / std::pow(i, a);
    }
    return numerator;
}

long double zipf_cdf(size_t x, size_t n, double a) {
    long double denominator = zipf_denominator(n, a);
    long double numerator = zipf_numerator(x, a);
    return numerator / denominator;
}


// Zipf Approximation from https://medium.com/@jasoncrease/zipf-54912d5651cc
static double approx_zipf_denominator(size_t n, double a) {
    return 12 * (std::pow(n, 1 - a) - 1) / (1 - a) + 6 + std::pow(n, -a) + a - a * std::pow(n, -1 - a);
}

static double approx_zipf_cdf(size_t x, double denominator, double a) {
    double numerator = 12 * (std::pow(x, 1 - a) - 1) / (1 - a) + 6 + std::pow(x, -a) + a - a * std::pow(x, -1 - a);
    return numerator / denominator;
}

static double approx_zipf_cdf(size_t x, size_t n, double a) {
    double denominator = approx_zipf_denominator(n, a);
    return approx_zipf_cdf(x, denominator, a);
}

static uint64 inverseCdfFast(double p, double s, double N) {

    double tolerance = 0.01;
    double x = (N - 1) / 2;

    double pD = p * (12 * (pow(N, -s + 1) - 1) / (1 - s) + 6 + 6 * pow(N, -s) + s - s * pow(N, -s - 1));

    while (true) {
        double m = pow(x, -s - 2);   // x ^ ( -s - 2)
        double mx = m * x;                // x ^ ( -s - 1)
        double mxx = mx * x;              // x ^ ( -s)
        double mxxx = mxx * x;            // x ^ ( -s + 1)

        double a = 12 * (mxxx - 1) / (1 - s) + 6 + 6 * mxx + s - (s * mx) - pD;
        double b = 12 * mxx - (6 * s * mx) + (m * s * (s + 1));
        double newx = std::max<double>(1, x - a / b);
        if (abs(newx - x) <= tolerance) {
            return std::min(round(newx), N - 1);
        }
        x = newx;
    }
}

#endif //STACKEDALTERNATINGAMQS_ZIPFDISTRIBUTION_H
