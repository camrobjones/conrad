# import statements
import numpy as np
import math
import conrad.perlin_noise_python_numpy as perlin
import matplotlib.cm as cm
import random
from scipy import ndimage


# Numbers



x_inds = np.array([[[x / 100, x / 100, x / 100] for x in range(-100, 100)] for i in range(200)])
y_inds = np.array([[[y / 100, y / 100, y / 100] for i in range(200)] for y in range(-100, 100)])
e_inds = np.array([[[math.e, math.e, math.e] for i in range(200)] for y in range(200)])
pi_inds = np.array([[[math.pi, math.pi, math.pi] for i in range(200)] for y in range(200)])


def r_inds():
    r3 = [random.random(), random.random(), random.random()]
    return np.array([[r3 for i in range(200)] for y in range(200)])


x_inds500 = np.array([[[x / 250, x / 250, x / 250] for x in range(-250, 250)] for i in range(500)])
y_inds500 = np.array([[[y / 250, y / 250, y / 250] for i in range(500)] for y in range(-250, 250)])
e_inds500 = np.array([[[math.e, math.e, math.e] for i in range(500)] for y in range(500)])
pi_inds500 = np.array([[[math.pi, math.pi, math.pi] for i in range(500)] for y in range(500)])

def r_inds500():
    r3 = [random.random(), random.random(), random.random()]
    return np.array([[r3 for i in range(500)] for y in range(500)])


x_inds800 = np.array([[[x / 400, x / 400, x / 400] for x in range(-400, 400)] for i in range(800)])
y_inds800 = np.array([[[y / 400, y / 400, y / 400] for i in range(800)] for y in range(-400, 400)])
e_inds800 = np.array([[[math.e, math.e, math.e] for i in range(800)] for y in range(800)])
pi_inds800 = np.array([[[math.pi, math.pi, math.pi] for i in range(800)] for y in range(800)])

def r_inds800():
    r3 = [random.random(), random.random(), random.random()]
    return np.array([[r3 for i in range(800)] for y in range(800)])

# x_inds_desktop = np.array([[[x/1920, x/1920, x/1920] for x in range(-960, 960)] for i in range(1080)])
# y_inds_desktop = np.array([[[y/1080, y/1080, y/1080] for i in range(1920)] for y in range(-540, 540)])
# e_inds_desktop = np.array([[[math.e, math.e, math.e] for i in range(1920)] for y in range(1080)])
# pi_inds_desktop = np.array([[[math.pi, math.pi, math.pi] for i in range(1920)] for y in range(1080)])
# r3 = [random.random(), random.random(), random.random()]
# r_inds_desktop = np.array([[r3 for i in range(1920)] for y in range(1080)])


num_dict = {'x': x_inds, 'y': y_inds, 'p': pi_inds, 'e': e_inds, 'r': r_inds()}
num_dict500 = {'x': x_inds500, 'y': y_inds500, 'p': pi_inds500, 'e': e_inds500, 'r': r_inds500()}
num_dict800 = {'x': x_inds800, 'y': y_inds800, 'p': pi_inds800, 'e': e_inds800, 'r': r_inds800()}


# num_dict_desktop = {'x':x_inds_desktop, 'y':y_inds_desktop, 'p':pi_inds_desktop, 'e':e_inds_desktop, 'r':r_inds_desktop}

def numbers(code, array_size=(200, 200)):
    if array_size == (200, 200):
        dict = num_dict
    elif array_size == (500, 500):
        dict = num_dict500
    elif array_size == (800, 800):
        dict = num_dict800
    elif array_size == (1920, 1080):
        dict = num_dict_desktop
    el0 = dict[code[0]]
    el1 = dict[code[1]]
    el2 = dict[code[2]]
    out = np.array(
        [[[el0[y][i][0], el1[y][i][1], el2[y][i][2]] for i in range(array_size[0])] for y in range(array_size[1])])
    return out

# Safe array functions

def invert(no):
    return (no * -1)


def myloge(no):
    no[no <= 0] = 0.001
    return np.log(no)


def mylog(no, base):
    no[no <= 0] = 0.001
    return np.log(no)


def myslog(no):
    if no <= 0:
        no = 0.001
    return math.log(no)


def mymod(no, div):
    div[div == 0] = 1
    return no % div


def mydiv(no, div):
    div[div == 0] = 1
    return no / div


def myminus(a, b):
    return a - b


def myproduct(a, b):
    return (a * b)


def myexp(a, b):
    try:
        b = np.round(b)
        a[np.bitwise_and(a <= 0, b <= 1)] = abs(a[np.bitwise_and(a <= 0, b <= 1)])
        a[a == 0] = 0.00001
        b[b > 10] = mylog(b[b > 10] * 2300, 10)
        a[a > 300] = mylog(a[a > 300] * 6.474754650804084e+127, 10)
        out = a ** b
        return out
    except:
        # print("EXP FAIL ON {0} ** {1}").format(a, b)
        print("EXP FAIL")
        return a ** 2


def mysum(a, b):
    return a + b


npround = np.vectorize(round)


def myround(a, b):
    if (b < 1).all():
        b = b * 10
    return npround(a, b.astype(int))

def by2(no):
    return no * 2


def by10(no):
    return no * 10


def by100(no):
    return no * 100


def mysqrt(no):
    return np.sqrt(abs(no))


def mynoise(no):
    return (1.0 * (no)) * random.random()


def myif(a, b, c, d):
    try:
        h, i, j, k = random.sample([a, b, c, d], 4)
        #h, i, j, k = a, b, c, d
        out = h
        out[j > i] = k[j > i]
        return out
    except ValueError:
        print(ValueError, a.shape, b.shape, c.shape, d.shape)
        return a
    return np.array(out)


def s_myif(a, b, c, d):
    h, i, j, k = random.sample([a, b, c, d], 4)
    #h, i, j, k = a, b, c, d
    return ("{0} if {1} > {2} else {3}".format(h, i, j, k))


def mymean(a, b):
    return (a + b) / 2


def mysquare(no):
    out = no
    out[no < 100000] = no[no < 100000] ** 2
    out[no >= 100000] = np.log(no[no >= 100000]) * 1087
    return out


def mymax(a, b):
    at = a.sum()
    bt = b.sum()
    if at >= bt:
        return a
    else:
        return b


def mymin(a, b):
    at = a.sum()
    bt = b.sum()
    if at <= bt:
        return a
    else:
        return b


def warp(array, array2):
    shift_func = random.choice([np.sin, np.cos, np.tan])
    k = random.choice([0.1, 0.5, 1, 2, 10])#[int(np.sum(array2[:, :, 0]) % 5)]
    A = array.shape[0] / 3.0
    w = 2.0 / array.shape[1]

    #shift = lambda x: A * shift_func(k * np.pi * x * w)
    shift = lambda x: shift_func(k * np.pi * x)

    for i in range(array.shape[0]):
        array[:, i] = np.roll(array[:, i], int(shift(i)))
    return array

def gradient(array, element=sum):
    out = np.gradient(array)
    return element(out)

def gradient0(array):
    out = np.gradient(array)
    return out[0]

def gradient1(array):
    out = np.gradient(array)
    return out[1]

def gradient2(array):
    out = np.gradient(array)
    return out[2]

def quick_convolve(array1):
    k = [[[1,1,1]]]
    out = ndimage.convolve(array1, k, mode='constant', cval=0.0)
    return out

def convolve(array1, array2):
    k = np.array([[[random.randint(0, 5) for x in range(3)]]])
    out = ndimage.convolve(array1, k, mode='constant', cval=0.0)
    return out

def radial(array, array2):
    fraction = random.choice([0.5, 0.25, 0.1, 0.05])#[int(np.sum(array2[:, :, 0]) % 4)]
    sx, sy, _ = array.shape
    X, Y = np.ogrid[-sx/2:sx/2, -sy/2:sy/2]
    X = X/(sx/2)
    Y = Y/(sy/2)
    r = np.hypot(X, Y)
    rbin = ((fraction) * r/r.max())

    cmap = 'hsv'
    rgba = cm.ScalarMappable(cmap='hsv').to_rgba(rbin)
    rgb = rgba[:, :, :3]
    out = array * rgb
    return out


def quick_perlin_noise(array, fraction=0.1, cmap='hsv'):
    size = array.shape[0]
    ns = int(fraction * size)
    noise = perlin.generate_2D_perlin_noise(size, ns)
    noisergba = cm.ScalarMappable(cmap=cmap).to_rgba(noise)
    noisergb = noisergba[:, :, :3]
    out = array * noisergb
    return out

def perlin_noise(array1, array2):
    size = array1.shape[0]
    fraction = random.choice([0.5, 0.25, 0.1, 0.05])#[int(np.sum(array2[:,:,0])%4)]
    ns = int(fraction * size)
    cmap = random.choice(['hsv', 'terrain', 'gist_rainbow', 'inferno'])#[int(np.sum(array2[:,:,1])%4)]
    noise = perlin.generate_2D_perlin_noise(size, ns)
    noisergba = cm.ScalarMappable(cmap=cmap).to_rgba(noise)
    noisergb = noisergba[:, :, :3]
    out = array1 * noisergb
    return out


# Read Gene Variables

num = {"AA": 'xxx', "AC": 'xxy', "AG": 'xyx', "AT": 'yxx',
       "CA": 'yyy', "CC": "yyx", "CG": "yxy", "CT": "xyy",
       "GA": 'xxr', "GC": "xry", "GG": "yxr", "GT": "ryy",
       "TA": 'per', "TC": "rep", "TG": "epe", "TT": "rrr"}

onearg = {"AA": abs, "AC": mysquare, "AG": invert, "AT": mynoise,
          "CA": gradient2, "CC": by10, "CG": by100, "CT": np.unwrap,
          "GA": np.sin, "GC": np.cos, "GG": np.tan, "GT": gradient1,
          "TA": myloge, "TC": quick_convolve, "TG": mysqrt, "TT": gradient0}

twoargs = {"AA": mysum, "AC": myround, "AG": mylog, "AT": radial,
           "CA": myproduct, "CC": myminus, "CG": myproduct, "CT": mydiv,
           "GA": mymod, "GC": perlin_noise, "GG": myexp, "GT": convolve,
           "TA": mymax, "TC": mymin, "TG": mymean, "TT": perlin_noise}

control = {"A": onearg, "C": twoargs, "G": myif, "T": twoargs}
first = {"A": onearg, "C": twoargs, "G": myif, "T": myif}

# Print Gene variables

s_num = {"AA": 'numbers("xxx")', "AC": 'numbers("xxy")', "AG": 'numbers("xyx")', "AT": 'numbers("yxx")',
       "CA": 'numbers("yyy")', "CC": "numbers('yyx')", "CG": "numbers('yxy')", "CT": "numbers('xyy')",
       "GA": 'numbers("xxr")', "GC": "numbers('xry')", "GG": "numbers('yxr')", "GT": "numbers('ryy')",
       "TA": 'numbers("per")', "TC": "numbers('rep')", "TG": "numbers('epe')", "TT": "numbers('rrr')"}

s_onearg = {"AA": 'abs', "AC": 'mysquare', "AG": 'invert', "AT": 'mynoise',
            "CA": 'gradient2', "CC": 'by10', "CG": 'by100', "CT": 'np.unwrap',
            "GA": 'np.sin', "GC": 'np.cos', "GG": 'np.tan', "GT": 'gradient1',
            "TA": 'myloge', "TC": 'quick_convolve', "TG": 'mysqrt', "TT": 'gradient0'}

s_twoargs = {"AA": 'mysum', "AC": 'myround', "AG": 'mylog', "AT": 'radial',
             "CA": 'convolve', "CC": 'myminus', "CG": 'myproduct', "CT": 'mydiv',
             "GA": 'mymod', "GC": 'perlin_noise', "GG": 'myexp', "GT": 'convolve',
             "TA": 'mymax', "TC": 'mymin', "TG": 'mymean', "TT": 'perlin_noise'}

s_control = {"A": s_onearg, "C": s_twoargs, "G": myif, "T": s_twoargs}
s_first = {"A": s_onearg, "C": s_twoargs, "G": myif, "T": myif}

# Read Genomes


def read_gene(genome, depth, acc, size):
    if depth >= (len(genome) - 1) - acc:
        return numbers(num[genome[acc:acc + 2]], size)
    elif depth == 0:
        d = first[genome[acc - 1]]
    else:
        d = control[genome[acc - 1]]

    if d == onearg:
        arg1 = read_gene(genome, depth + 3, acc + 3, size)
        return d[genome[acc:acc + 2]](arg1)
    elif d == twoargs:
        return d[genome[acc:acc + 2]](read_gene(genome, depth + 3, acc + 3, size),
                                      read_gene(genome, depth + 6, acc + 6, size))
    elif d == num:
        return numbers(num[genome[acc:acc + 2]], size)
    else:
        return myif(read_gene(genome, depth + 3, acc + 3, size), read_gene(genome, depth + 6, acc + 6, size),
                    read_gene(genome, depth + 9, acc + 9, size), read_gene(genome, depth + 12, acc + 12, size))




def print_gene(genome, depth, acc):
    if depth >= (len(genome) - 1) - acc:
        return s_num[genome[acc:acc + 2]]
    elif depth == 0:
        d = s_first[genome[acc - 1]]
    else:
        d = s_control[genome[acc - 1]]
    if d == s_onearg:
        return "{0}({1})".format(d[genome[acc:acc + 2]], str(print_gene(genome, depth + 3, acc + 3)))
    elif d == s_twoargs:
        return "{0}({1}, {2})".format(d[genome[acc:acc + 2]], print_gene(genome, depth + 3, acc + 3),
                                      print_gene(genome, depth + 6, acc + 6))
    elif d == num:
        return s_num[genome[acc:acc + 2]]
    else:
        return s_myif(print_gene(genome, depth + 3, acc + 3), print_gene(genome, depth + 6, acc + 6),
                    print_gene(genome, depth + 9, acc + 9), print_gene(genome, depth + 12, acc + 12))


