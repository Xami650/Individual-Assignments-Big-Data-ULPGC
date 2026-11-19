package org.ulpgc.bd.matmul.dist;

import com.hazelcast.client.HazelcastClient;
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.IntFunction;
import org.ulpgc.bd.matmul.kernel.BandKernel;

/**
 * An embedded cluster for the tests: w Hazelcast members in the test JVM (the
 * configuration of the real members, {@link HazelcastSetup#memberConfig}) and a
 * client with the driver's configuration. The benchmark runs every member in its
 * own process instead (scripts/cluster.py); scripts/verify.py checks that path.
 */
public final class TestCluster implements AutoCloseable {

    /** Ports far from the benchmark's (5901 + ...), a new range for every cluster of the test run. */
    private static int nextBasePort = 7101 + 20 * (int) (ProcessHandle.current().pid() % 40);

    public final String name;
    public final int basePort;
    final List<HazelcastInstance> members = new ArrayList<>();
    final List<BandKernel> kernels = new ArrayList<>();
    public final HazelcastInstance client;
    public final DistributedMatMul dist;

    TestCluster(int workers, IntFunction<BandKernel> kernel) throws Exception {
        HazelcastSetup.quietLogging();
        name = "a4-test-" + UUID.randomUUID();
        synchronized (TestCluster.class) {
            basePort = nextBasePort;
            nextBasePort += 20;
        }
        for (int m = 0; m < workers; m++) {
            kernels.add(kernel.apply(m));
        }
        // Each member's start blocks until all have joined (initial.min.cluster.size), so start them together.
        ExecutorService starter = Executors.newFixedThreadPool(workers);
        try {
            List<Future<HazelcastInstance>> started = new ArrayList<>();
            for (int m = 0; m < workers; m++) {
                int index = m;
                started.add(starter.submit(() -> Hazelcast.newHazelcastInstance(
                        HazelcastSetup.memberConfig(name, basePort, workers, index, kernels.get(index)))));
            }
            for (Future<HazelcastInstance> f : started) {
                members.add(f.get());
            }
        } finally {
            starter.shutdown();
        }
        members.get(0).getPartitionService().getPartition(0).getOwner();
        long deadline = System.nanoTime() + Duration.ofSeconds(60).toNanos();
        while (!members.get(0).getPartitionService().isClusterSafe()) {
            if (System.nanoTime() > deadline) {
                throw new IllegalStateException("cluster not safe");
            }
            Thread.sleep(10);
        }
        client = HazelcastClient.newHazelcastClient(HazelcastSetup.clientConfig(name, basePort, workers));
        dist = new DistributedMatMul(client, workers, Duration.ofSeconds(60));
    }

    public TestCluster(int workers, int threads) throws Exception {
        this(workers, m -> new BandKernel(threads));
    }

    /** Replaces the kernel of every member (and closes the old ones). */
    void useKernels(IntFunction<BandKernel> kernel) {
        for (int m = 0; m < members.size(); m++) {
            BandKernel next = kernel.apply(m);
            members.get(m).getUserContext().put(HazelcastSetup.CONTEXT_KERNEL, next);
            kernels.get(m).close();
            kernels.set(m, next);
        }
    }

    @Override
    public void close() {
        client.shutdown();
        for (HazelcastInstance m : members) {
            m.getLifecycleService().terminate();
        }
        kernels.forEach(BandKernel::close);
    }
}
