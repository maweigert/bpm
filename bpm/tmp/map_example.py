import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
from time import time




KERNEL = """
__kernel void matvec(
  __global const float *a_g,
  __global const float *x_g,
  __global const float *b_g,
  __global float *y_g,
  unsigned int mat_width)
{
    uint i = get_global_id(0);
    uint mat_height = get_global_size(0);

    float result = b_g[i];
    for (int j = 0; j < mat_width; ++j)
        result += a_g[i + j*mat_height] * x_g[j];

    y_g[i] = result;
}
"""



use_profiling = 0

if use_profiling:
    cq_props = cl.command_queue_properties.PROFILING_ENABLE
else:
    cq_props = 0




class MCRunner:
    def __init__(self, ctx, a_dev, b_dev, mat_vec_knl):
        self.a_dev = a_dev
        self.b_dev = b_dev
        self.context = ctx
        self.mat_vec_knl = mat_vec_knl

        queue = self.queue = cl.CommandQueue(ctx, properties=cq_props)

        mat_shape = a_dev.shape

        self.x_dev = cl_array.empty(queue, (mat_shape[1],),
                dtype=np.float32)
        mf = cl.mem_flags
        self.y_host_buf = cl.Buffer(ctx, mf.ALLOC_HOST_PTR, self.b_dev.nbytes)
        self.y_host = self.y_host_buf.get_host_array(
                mat_shape[0], dtype=np.float32)

    def start(self):
        mat_shape = self.a_dev.shape

        rand_start_t = time()
        self.x_host = np.random.uniform(size=mat_shape[1]).astype(np.float32)
        self.rand_t = time()-rand_start_t

        self.x_wr_evt = cl.enqueue_write_buffer(
                self.queue, self.x_dev.data, self.x_host, is_blocking=False)
        self.mv_evt = self.mat_vec_knl(self.queue, (mat_shape[0],), (128,),
                self.a_dev.data, self.x_dev.data, self.b_dev.data, self.y_host_buf,
                mat_shape[1])

    def finish(self, tmg):
        self.mv_evt.wait()

        if use_profiling:
            tmg.rand_t += self.rand_t
            tmg.x_write_t += 1e-9*(self.x_wr_evt.profile.END-self.x_wr_evt.profile.START)
            tmg.matvec_t += 1e-9*(self.mv_evt.profile.END-self.mv_evt.profile.START)

        norm_start_t = time()
        result = la.norm(self.y_host)**2
        tmg.norm_t += time() - norm_start_t

        return result

def main():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cq_props)

    sample_count = 100
    mat_shape = (1024*1024, 32)

    a_host = np.asarray(
            np.random.randn(*mat_shape),
            dtype=np.float32, order="F")

    a_dev = cl_array.to_device(queue, a_host)
    b_host = np.random.randn(mat_shape[0]).astype(np.float32)
    b_dev = cl_array.to_device(queue, b_host)

    mat_vec_prg = cl.Program(ctx, KERNEL).build()
    mat_vec_knl = mat_vec_prg.matvec
    mat_vec_knl.set_scalar_arg_dtypes(
            [None, None, None, None, np.uint32])

    norms = []

    class Timing:
        pass

    tmg = Timing()
    tmg.rand_t = 0
    tmg.x_write_t = 0
    tmg.y_read_t = 0
    tmg.matvec_t = 0
    tmg.y_read_t = 0
    tmg.norm_t = 0

    avl_runners = [
            MCRunner(ctx, a_dev, b_dev, mat_vec_knl)
            for i in range(3)]
    busy_runners = []

    print "enter sample loop"

    total_start_t = time()

    for sample in xrange(sample_count):
        if avl_runners:
            rnr = avl_runners.pop(0)
        else:
            rnr = busy_runners.pop(0)
            norms.append(rnr.finish(tmg))

        rnr.start()
        busy_runners.append(rnr)

    while busy_runners:
        rnr = busy_runners.pop(0)
        norms.append(rnr.finish(tmg))

    total_end_t = time()

    elapsed = total_end_t - total_start_t

    from pytools import string_histogram
    print string_histogram(np.array(norms)/mat_shape[0], bin_count=30,
            use_unicode=False)

    print

    print "total: %.2e s" % (elapsed/sample_count)
    if use_profiling:
        elapsed_from_parts = (
                tmg.rand_t + tmg.x_write_t + tmg.matvec_t
                + tmg.norm_t + tmg.y_read_t + tmg.norm_t)

        print "total (from parts): %.2e s" % (elapsed_from_parts/sample_count)
        print "rand: %.2e s" % (tmg.rand_t/sample_count)
        print "x write: %.2e s" % (tmg.x_write_t/sample_count)
        print "matvec: %.2e s" % (tmg.matvec_t/sample_count)
        print "y read: %.2e s" % (tmg.y_read_t/sample_count)
        print "norm: %.2e s" % (tmg.norm_t/sample_count)




if __name__ == "__main__":
    main()
