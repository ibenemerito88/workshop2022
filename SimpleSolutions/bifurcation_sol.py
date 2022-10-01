import numpy as np
import matplotlib.pyplot as plt

import reslast


plt.close("all")

# Radius: bifurcation

q111, a111, p111, u111, c111, names111, t111 = reslast.resu("bifurcation_R111")
q112, a112, p112, u112, c112, names112, t112 = reslast.resu("bifurcation_R112")

plt.figure()
plt.plot(q111["1-P"][:,3],label="r=r0")
plt.plot(q112["1-P"][:,3],label="r=r0")
plt.legend()
plt.title("Radius: bifurcation - Flow in 1-P")

plt.figure()
plt.plot(q111["2-d1"][:,3],label='r=r0')
plt.plot(q112["2-d1"][:,3],label='r=r0')
plt.legend()
plt.title("Radius: bifurcation - Flow in 2-d1")

plt.figure()
plt.plot(q111["3-d2"][:,3],label='r=r0')
plt.plot(q112["3-d2"][:,3],label='r=r1>r0')
plt.title("Radius: bifurcation - Flow in 3-d2")
plt.legend()



plt.figure()
plt.plot(p111["1-P"][:,3],label="r=r0")
plt.plot(p112["1-P"][:,3],label="r=r0")
plt.legend()
plt.title("Radius: bifurcation - Pressure in 1-P")

plt.figure()
plt.plot(p111["2-d1"][:,3],label='r=r0')
plt.plot(p112["2-d1"][:,3],label='r=r0')
plt.legend()
plt.title("Radius: bifurcation - Pressure in 2-d1")

plt.figure()
plt.plot(p111["3-d2"][:,3],label='r=r0')
plt.plot(p112["3-d2"][:,3],label='r=r1>r0')
plt.title("Radius: bifurcation - Pressure in 3-d2")
plt.legend()


plt.show()



