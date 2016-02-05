"""
test whether async memory transfer can help....
"""


from gputools import get_device,  OCLArray, OCLElementwiseKernel
from time import time
import numpy as np
from pyopencl import array
import pyopencl as cl


fill_func =  OCLElementwiseKernel("float *a_g, const int niter",
"""float res = a_g[i];
for (int i = 0;i<niter;++i)
    res += 1./(i+1)/(i+1);
a_g[i] = res;
""",
"name")



# def copy_arr(a,a_g, blocking  = True, async = False):


def estimate_niter(N):
    """returns niter s.t. the time spent on kernel is same as for memory transfer"""
    a = np.ones(N,np.float32)


    dev = get_device()
    context, queue = dev.context, dev.queue
    mf = cl.mem_flags


    t = time()
    copy_g = cl.Buffer(context, mf.ALLOC_HOST_PTR
                       ,size = a.nbytes)

    cl.enqueue_map_buffer(queue, copy_g, a,
                    device_offset=0,
                    is_blocking=False)


    # cl.enqueue_copy(queue, copy_g, a,
    #                 device_offset=0,
    #                 is_blocking=False)

    queue.flush()

    # a_g = OCLArray.from_array(a, async = True)
    #a_g = array.to_device(queue, a, async = False)
    print time()-t

    # return a_g.get()





if __name__ == '__main__':


    # ctx = cl.create_some_context()
    # queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    dev = get_device()
    ctx, queue = dev.context, dev.queue


    data = np.zeros((512, 512, 200), dtype=np.float32)
    data_cl = cl. Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=data.nbytes)
    #data_cl = cl. Buffer(ctx, cl.mem_flags.READ_WRITE, size=data.nbytes)
    cl.enqueue_copy(queue, data_cl, data)
    queue.finish()
    t_start = time()
    evt = cl.enqueue_copy(queue, data_cl, data, is_blocking=False)
    t_wait = time()
    evt.wait()
    t_end = time()
    print "copy time=", t_wait - t_start
    print "Wait time", t_end - t_wait
    print "Profile time", 1e-9 * (evt.profile.end - evt.profile.start)

    # data = np.ones(2**29,np.int8)
    #
    # dev = get_device()
    # context, queue = dev.context, dev.queue
    # mf = cl.mem_flags
    # mapf = cl.map_flags
    #
    # copy_g = cl.Buffer(context, mf.ALLOC_HOST_PTR
    #                    ,size = data.nbytes)
    #
    #
    # t = time()
    #
    # cl.enqueue_write_buffer(queue, copy_g, data,
    #                 device_offset=0,
    #                 is_blocking=True)
    #
    # print time()-t

    # data = np.ones(2**28,np.int8)
    #
    # dev = get_device()
    # context, queue = dev.context, dev.queue
    # mf = cl.mem_flags
    # mapf = cl.map_flags
    #
    # dataSize  = data.nbytes
    #
    # pinInBuffer = cl.Buffer(context, mf.READ_ONLY|
    #                     mf.ALLOC_HOST_PTR, dataSize)
    # pinOutBuffer = cl.Buffer(context, mf.WRITE_ONLY|
    #                      mf.ALLOC_HOST_PTR, dataSize)
    #
    #
    # devInBuffer = cl.Buffer(context, mf.READ_ONLY, dataSize)
    # devOutBuffer = cl.Buffer(context, mf.WRITE_ONLY, dataSize)
    #
    # # Get numpy arrays used for filling and retrieving data from
    # # pinned-memory
    # (dataIn,ev) = cl.enqueue_map_buffer(queue, pinInBuffer,
    #                                 mapf.WRITE,
    #                                 0, (dataSize,), np.int8, 'C')
    # (dataOut,ev) = cl.enqueue_map_buffer(queue, pinOutBuffer,
    #                                  mapf.READ,
    #                                  0, (dataSize,), np.int8, 'C')
    #
    #
    #
    #
    # # Fill the array obtained from memory maps
    # dataIn[:] = np.frombuffer(data, dtype=np.uint8)
    #
    # t = time()
    #
    # for _ in xrange(10):
    #     cl.enqueue_copy(queue, devInBuffer, dataIn,
    #             is_blocking=False)
    #
    # #queue.flush()
    # print time()-t
# self.cmdQueues[0].flush()
#
# # Launch kernel on the first half
# program.aes_ecb(self.cmdQueues[0], (halfSize>>4,), (256,), keyBuffer,
#                 devInBuffer, devOutBuffer,
#                 T0buff, T1buff, T2buff, T3buff,
#                 np.uint32(0))
#
# # Start copying the second half
# cl.enqueue_copy(self.cmdQueues[1], devInBuffer,
#                 dataIn[halfSize-roundoffSize:],
#                 device_offset=halfSize-roundoffSize, is_blocking=False)
#
# self.cmdQueues[0].flush()
# self.cmdQueues[1].flush()
#
# # Launch kernel on the second half
# program.aes_ecb(self.cmdQueues[1], (halfSize>>4,), (256,), keyBuffer,
#                 devInBuffer, devOutBuffer,
#                 T0buff, T1buff, T2buff, T3buff,
#                 np.uint32(halfSize>>4))
#
# # Non-blocking read of the first half
# cl.enqueue_copy(self.cmdQueues[0], dataOut[:halfSize], devOutBuffer,
#                 is_blocking=False)
#
# self.cmdQueues[0].flush()
# self.cmdQueues[1].flush()
#
# # Finally, read the second half
# cl.enqueue_copy(self.cmdQueues[1], dataOut[halfSize-roundoffSize:],
#                 devOutBuffer,
#                 device_offset=halfSize-roundoffSize)
