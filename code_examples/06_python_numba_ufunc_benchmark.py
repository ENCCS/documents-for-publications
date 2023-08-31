#
# python
#
import numpy as np
x = np.random.rand(10000000)
res = np.random.rand(10000000)

%%timeit -r 1
for i in range(10000000):
    res[i]=f(x[i], x[i])
# 6.75 s $\pm$ 0 ns per loop (mean $\pm$ std. dev. of 1 run, 1 loop each)





#
# Numba cpu
#
import numpy as np
import numba

x = np.random.rand(10000000)
res = np.random.rand(10000000)

%timeit res=f_numba_cpu(x, x)
# 734 ms $\pm$ 435 $\mu$s per loop (mean $\pm$ std. dev. of 7 runs, 1 loop each)





#
# Numba gpu
#
import numpy as np
import numba

x = np.random.rand(10000000)
res = np.random.rand(10000000)

%timeit res=f_numba_gpu(x, x)
# 78.4 ms $\pm$ 6.71 ms per loop (mean $\pm$ std. dev. of 7 runs, 1 loop each)
