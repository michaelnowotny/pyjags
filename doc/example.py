import pyjags
import numpy as np

np.random.seed(0)
np.set_printoptions(precision=1)

N = 500
a = 70
b = 4
sigma = 50

x = np.random.uniform(0, 100, size=N)
y = np.random.normal(a + x * b, sigma, size=N)

code = """
model {
    for (i in 1:N) {
        y[i] ~ dnorm(alpha + beta * x[i], tau)
    }
    alpha ~ dunif(-1e3, 1e3)
    beta ~ dunif(-1e3, 1e3)
    tau <- 1 / sigma^2
    sigma ~ dgamma(1e-4, 1e-4)
}
"""

model = pyjags.Model(code, data=dict(x=x, y=y, N=N), chains=4)
samples = model.sample(5000, vars=["alpha", "beta", "sigma"])


def summary(samples, varname, p=95):
    values = samples[varname]
    ci = np.percentile(values, [100 - p, p])
    print(
        "{:<6} mean = {:>5.1f}, {}% credible interval [{:>4.1f} {:>4.1f}]".format(
            varname, np.mean(values), p, *ci
        )
    )


for varname in ["alpha", "beta", "sigma"]:
    summary(samples, varname)
