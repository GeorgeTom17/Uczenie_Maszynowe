import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, Math, Latex

data = pd.read_csv("fires_thefts.csv", names=["x", "y"])
x = data["x"].to_numpy()
y = data["y"].to_numpy() #zbieranie danych


def regdots(x, y):
    fig = plt.figure(figsize=(16 * 0.6, 9 * 0.6))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax.scatter(x, y, c="r", label="Dane")

    ax.set_xlabel("Pożary")
    ax.set_ylabel("Włamania")
    ax.margins(0.05, 0.05)
    plt.ylim(min(y) - 5, max(y) + 5)
    plt.xlim(min(x) - 5, max(x) + 5)
    return fig #tworzenie wykresu z danych


def regline(fig, fun, theta, x):
    ax = fig.axes[0]
    x0, x1 = min(x), max(x)
    X = [x0, x1]
    Y = [fun(theta, x) for x in X]
    ax.plot(
        X,
        Y,
        linewidth="2",
        label=(
            r"$y={theta0}{op}{theta1}x$".format(
                theta0=theta[0],
                theta1=(theta[1] if theta[1] >= 0 else -theta[1]),
                op="+" if theta[1] >= 0 else "-",
            )
        ),
    )


def legend(fig):
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    # try-except block is a fix for a bug in Poly3DCollection
    try:
        fig.legend(handles, labels, fontsize="15", loc="lower right")
    except AttributeError:
        pass


def h(theta, x):
    return theta[0] + theta[1] * x


def J(h, theta, x, y):
    m = len(y)
    return 1.0 / (2 * m) * sum((h(theta, x[i]) - y[i]) ** 2 for i in range(m))


def regline2(fig, fun, theta, xx, yy):
    """Rysuj regresję liniową"""
    ax = fig.axes[0]
    x0, x1 = min(xx), max(xx)
    X = [x0, x1]
    Y = [fun(theta, x) for x in X]
    cost = J(fun, theta, xx, yy)
    ax.plot(
        X,
        Y,
        linewidth="2",
        label=(
            r"$y={theta0}{op}{theta1}x, \; J(\theta)={cost:.3}$".format(
                theta0=theta[0],
                theta1=(theta[1] if theta[1] >= 0 else -theta[1]),
                op="+" if theta[1] >= 0 else "-",
                cost=cost,
            )
        ),
    )


sliderTheta01 = widgets.FloatSlider(
    min=-10, max=10, step=0.1, value=0, description=r"$\theta_0$", width=300
)
sliderTheta11 = widgets.FloatSlider(
    min=-5, max=5, step=0.1, value=0, description=r"$\theta_1$", width=300
)

sliderTheta02 = widgets.FloatSlider(
    min=-10, max=10, step=0.1, value=0, description=r"$\theta_0$", width=300
)
sliderTheta12 = widgets.FloatSlider(
    min=-5, max=5, step=0.1, value=0, description=r"$\theta_1$", width=300
)


def slide2(theta0, theta1):
    fig = regdots(x, y)
    regline2(fig, h, [theta0, theta1], x, y)
    legend(fig)

def slide1(theta0, theta1):
    fig = regdots(x, y)
    regline(fig, h, [theta0, theta1], x)
    legend(fig)
    plt.show()
