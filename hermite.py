import math
import sympy as sp
import numpy as np


def takeFirst(elem):
    return elem[0]


def interpolate_hermite(data):
    alldata = []
    for i in range(len(data)):
        alldata = alldata + data[i]
    alldata.sort(key=takeFirst)
    p = []
    size = len(alldata)
    for i in range(size):
        p.append([0] * (size + 1))
    for i in range(size):
        p[i][0] = alldata[i][0]
    p[0][1] = alldata[0][1]
    for i in range(1, size):
        if p[i][0] == p[i - 1][0]:
            p[i][1] = p[i - 1][1]
        else:
            p[i][1] = alldata[i][1]

    for k in range(2, size + 1):
        for i in range(size):
            if i + k <= size:
                if (p[i][k - 1] - p[i + 1][k - 1]) == 0 and (p[i][0] - p[i + k - 1][0]) == 0:
                    search = data[k - 1]
                    for b in search:
                        if b[0] == p[i][0]:
                            p[i][k] = b[1] / math.factorial(k - 1)
                            break
                else:
                    p[i][k] = (p[i][k - 1] - p[i + 1][k - 1]) / (p[i][0] - p[i + k - 1][0])
    return p


def matrix_to_polynom(matrix):
    x = sp.Symbol('x')
    f = sp.Function('z')(x)
    f += matrix[0][1]
    size = len(matrix)
    for i in range(2, size + 1):
        zeroing = 1
        for a in range(i - 1):
            zeroing = zeroing * (x - matrix[a][0])
        poly = matrix[0][i] * zeroing
        f += poly
    f -= sp.Function('z')(x)
    return sp.simplify(f)


# dataSet1 = [[(1,2),(3,5)],[(1,1),(3,-1)]]
# dataSet2 = [[(1,2),(3,5),(6,7)],[(1,1),(3,-1),(6,2)]]
dataSet3 = [[(0, 1), (1, 2)], [(0, 0), (1, 5)], [(0, 0), (1, 20)]]
# dataSet4 = [[(1,1),(2,3)],[(1,4),(2,1)],[(2,2)]]
dataSet5 = [[(0, 1.0), (1, 2.718281828459045), (2, 7.38905609893065),
             (3, 20.085536923187668), (4, 54.598150033144236),
             (5, 148.4131591025766), (6, 403.4287934927351),
             (7, 1096.6331584284585), (8, 2980.9579870417283),
             (9, 8103.083927575384)]]
dataSet6 = [[(0, 0), (2 * math.pi, 0)]]

dataSet7 = [[(0, 0), (2 * math.pi, 0)],
            [(0, 1), (2 * math.pi, 1)],
            [(0, 0), (2 * math.pi, 0)],
            [(0, -1), (2 * math.pi, -1)]]

f_hermite = matrix_to_polynom(interpolate_hermite(dataSet5))

x = sp.Symbol('x')
xs = np.linspace(0, 9, 100)
ys = [sp.exp(x).subs(x, xi) for xi in xs]

p1 = sp.plot(
    f_hermite, xlim=[-0.1, 9.0], ylim=[-0.1, 9000.0], show=False,
    markers=[{'args': [xs, ys, 'ro'], 'ms': 2}]
)

# p2 = sp.plot(sp.exp(x), show=False, line_color='green')
# p1.append(p2[0])

p1.show()

g_hermite = matrix_to_polynom(interpolate_hermite(dataSet3))
xs = np.linspace(0, 1, 2)
ys = [g_hermite.subs(x, xi) for xi in xs]

p3 = sp.plot(
    g_hermite, xlim=[-0.1, 1.1], ylim=[-0.1, 2.1], show=False,
    markers=[{'args': [xs, ys, 'ro'], 'ms': 4}]
    )

p3.show()

h_hermite = matrix_to_polynom(interpolate_hermite(dataSet6))
xs = np.linspace(0, 2 * math.pi, 2)
ys = [h_hermite.subs(x, xi) for xi in xs]

p4 = sp.plot(
    h_hermite, xlim=[-0.1, 2 * math.pi], ylim=[-1.1, 1.1], show=False,
    markers=[{'args': [xs, ys, 'ro'], 'ms': 4}]
    )
p4_sin = sp.plot(sp.sin(x), show=False, line_color='green')
p4.append(p4_sin[0])

p4.show()

i_hermite = matrix_to_polynom(interpolate_hermite(dataSet7))
xs = np.linspace(0, 2 * math.pi, 2)
ys = [i_hermite.subs(x, xi) for xi in xs]

p5 = sp.plot(
    i_hermite, xlim=[-0.1, 2 * math.pi], ylim=[-1.1, 1.1], show=False,
    markers=[{'args': [xs, ys, 'ro'], 'ms': 4}]
    )
p4_sin = sp.plot(sp.sin(x), show=False, line_color='green')
p5.append(p4_sin[0])

p5.show()
