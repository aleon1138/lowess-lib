#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include "../inc/nelder_mead.h"

// f(x,y) = x^2 + y^2, minimum at (0, 0) with f=0
static double quadratic(const std::array<double, 2> &p)
{
    return p[0]*p[0] + p[1]*p[1];
}

// f(x,y) = (x-3)^2 + 2*(y+1)^2, minimum at (3, -1) with f=0
static double off_center(const std::array<double, 2> &p)
{
    double dx = p[0] - 3.0;
    double dy = p[1] + 1.0;
    return dx*dx + 2.0*dy*dy;
}

// Rosenbrock banana: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1) with f=0
static double rosenbrock(const std::array<double, 2> &p)
{
    double a = 1.0 - p[0];
    double b = p[1] - p[0]*p[0];
    return a*a + 100.0*b*b;
}

TEST(NelderMead, Quadratic)
{
    // step=0.5 is half the distance from start to minimum; avoids restart
    auto r = nelder_mead<double, 2>(quadratic, {1.0, 1.0}, 1e-12, {0.5, 0.5});
    EXPECT_EQ(r.ifault, 0);
    EXPECT_NEAR(r.xmin[0], 0.0, 1e-4);
    EXPECT_NEAR(r.xmin[1], 0.0, 1e-4);
    EXPECT_LT(r.ynewlo, 1e-8);
}

TEST(NelderMead, OffCenter)
{
    // step matches the distance from start {0,0} to minimum {3,-1} so that
    // the initial simplex is well-scaled and avoids a spurious restart
    auto r = nelder_mead<double, 2>(off_center, {0.0, 0.0}, 1e-12, {3.0, 1.0});
    EXPECT_EQ(r.ifault, 0);
    EXPECT_NEAR(r.xmin[0],  3.0, 1e-4);
    EXPECT_NEAR(r.xmin[1], -1.0, 1e-4);
    EXPECT_LT(r.ynewlo, 1e-8);
}

TEST(NelderMead, Rosenbrock)
{
    auto r = nelder_mead<double, 2>(rosenbrock, {0.0, 0.0}, 1e-18, {0.5, 0.5});
    EXPECT_EQ(r.ifault, 0);
    EXPECT_NEAR(r.xmin[0], 1.0, 1e-3);
    EXPECT_NEAR(r.xmin[1], 1.0, 1e-3);
    EXPECT_LT(r.ynewlo, 1e-6);
}

TEST(NelderMead, KcountLimit)
{
    // With only 5 evaluations the optimizer cannot converge — expect ifault==2
    auto r = nelder_mead<double, 2>(rosenbrock, {0.0, 0.0}, 1e-18, {0.5, 0.5}, 1, 5);
    EXPECT_EQ(r.ifault, 2);
}

TEST(NelderMead, BadReqmin)
{
    // reqmin <= 0 is invalid — expect ifault==1
    auto r = nelder_mead<double, 2>(quadratic, {1.0, 1.0}, 0.0, {1.0, 1.0});
    EXPECT_EQ(r.ifault, 1);
}
