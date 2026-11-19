package org.ulpgc.bd.matmul.dist;

import com.hazelcast.client.config.ClientConfig;
import com.hazelcast.client.config.ClientConnectionStrategyConfig.ReconnectMode;
import com.hazelcast.config.Config;
import com.hazelcast.config.ExecutorConfig;
import com.hazelcast.config.InMemoryFormat;
import com.hazelcast.config.JoinConfig;
import com.hazelcast.config.MapConfig;
import com.hazelcast.config.NetworkConfig;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.ulpgc.bd.matmul.dist.protocol.MatMulSerialization;
import org.ulpgc.bd.matmul.kernel.BandKernel;

/**
 * The Hazelcast configuration of the cluster members and of the driver's
 * client, in one place (it is also printed in the report). Everything that is
 * not set here keeps Hazelcast 5.7's default.
 *
 * <p>Local execution: all processes run on one machine and talk over TCP on
 * the loopback interface (127.0.0.1). Member m listens on port base + m; the
 * members find each other through the fixed list of these addresses (TCP/IP
 * join, multicast and cloud auto-detection off), under a cluster name that is
 * new for every configuration, so that they cannot join another cluster. The
 * base port 5901 avoids Hazelcast's default 5701.
 */
public final class HazelcastSetup {

    /** Map of the bands of A: band t under key t, stored on member t. */
    public static final String MAP_A = "matmul-A";
    /** Map of the copies of B: one complete copy under key t for every member t. */
    public static final String MAP_B = "matmul-B";
    /** Map of the bands of C, written by the tasks and read by the driver. */
    public static final String MAP_C = "matmul-C";
    /** Executor service that runs the band tasks. */
    public static final String EXECUTOR = "matmul";
    /** User-context entry holding the member's {@link BandKernel}. */
    public static final String CONTEXT_KERNEL = "matmul.kernel";
    /** User-context entry holding the member's index (0 .. w-1). */
    public static final String CONTEXT_INDEX = "matmul.member-index";
    /** Member attribute with the member's index, visible to the client. */
    public static final String ATTRIBUTE_INDEX = "matmul.index";
    public static final int DEFAULT_BASE_PORT = 5901;
    public static final String LOOPBACK = "127.0.0.1";

    /** Strong reference, so that the level set by {@link #quietLogging()} is kept. */
    private static final Logger HAZELCAST_LOGGER = Logger.getLogger("com.hazelcast");

    private HazelcastSetup() {
    }

    /** Hazelcast logs through java.util.logging; only warnings and errors are printed. */
    public static void quietLogging() {
        HAZELCAST_LOGGER.setLevel(Level.WARNING);
    }

    /** Addresses of the w members: 127.0.0.1:base .. 127.0.0.1:base + w - 1. */
    public static List<String> addresses(int basePort, int members) {
        List<String> list = new ArrayList<>(members);
        for (int m = 0; m < members; m++) {
            list.add(LOOPBACK + ":" + (basePort + m));
        }
        return list;
    }

    /**
     * Configuration of member {@code index} of a cluster of {@code members}
     * members, whose band tasks run on {@code kernel}.
     */
    public static Config memberConfig(String cluster, int basePort, int members, int index, BandKernel kernel) {
        Config config = new Config();
        config.setClusterName(cluster);
        config.setInstanceName(cluster + "-member-" + index);
        // No usage statistics are sent to Hazelcast (the "phone home" call is on by default).
        config.setProperty("hazelcast.phone.home.enabled", "false");
        config.setProperty("hazelcast.logging.type", "jdk");
        // Hazelcast.newHazelcastInstance returns only once all members have joined.
        config.setProperty("hazelcast.initial.min.cluster.size", String.valueOf(members));
        // Bind only to the loopback interface.
        config.setProperty("hazelcast.socket.bind.any", "false");

        NetworkConfig network = config.getNetworkConfig();
        // SO_REUSEADDR: a port can be bound again while connections of an earlier cluster are in TIME_WAIT.
        network.setPort(basePort + index).setPortAutoIncrement(false).setReuseAddress(true);
        network.getInterfaces().setEnabled(true).addInterface(LOOPBACK);
        JoinConfig join = network.getJoin();
        join.getMulticastConfig().setEnabled(false);
        join.getAutoDetectionConfig().setEnabled(false);
        join.getTcpIpConfig().setEnabled(true).setMembers(addresses(basePort, members));

        // Background metric collection (every 5 s by default) is not used; the
        // driver reads the byte counters of its own connections instead.
        config.getMetricsConfig().setEnabled(false);
        config.getJetConfig().setEnabled(false);

        // No backup copies: every entry exists once, on the member that owns its partition
        // (no fault tolerance, declared in the report).
        for (String name : List.of(MAP_A, MAP_B, MAP_C)) {
            config.addMapConfig(new MapConfig(name).setBackupCount(0).setAsyncBackupCount(0)
                    .setInMemoryFormat(InMemoryFormat.BINARY).setStatisticsEnabled(false));
        }
        // One executor thread per member: each member receives one band task per multiplication,
        // and the task's computation runs on the kernel's k threads.
        config.addExecutorConfig(new ExecutorConfig(EXECUTOR).setPoolSize(1).setStatisticsEnabled(false));

        config.getSerializationConfig().addDataSerializableFactory(MatMulSerialization.FACTORY_ID,
                new MatMulSerialization());
        config.getMemberAttributeConfig().setAttribute(ATTRIBUTE_INDEX, String.valueOf(index));
        config.getUserContext().put(CONTEXT_KERNEL, kernel);
        config.getUserContext().put(CONTEXT_INDEX, index);
        return config;
    }

    /** Configuration of the driver's client for a cluster of {@code members} members. */
    public static ClientConfig clientConfig(String cluster, int basePort, int members) {
        ClientConfig config = new ClientConfig();
        config.setClusterName(cluster);
        config.setProperty("hazelcast.logging.type", "jdk");
        // A band task on one member can run for minutes at the largest sizes.
        config.setProperty("hazelcast.client.invocation.timeout.seconds", "900");
        config.getNetworkConfig().setAddresses(addresses(basePort, members));
        // Default routing (all members): the client opens a connection to every member and
        // sends each key-based operation directly to the member that owns the key.
        config.getConnectionStrategyConfig().setAsyncStart(false).setReconnectMode(ReconnectMode.OFF)
                .getConnectionRetryConfig().setClusterConnectTimeoutMillis(60_000);
        config.getMetricsConfig().setEnabled(false);
        config.getSerializationConfig().addDataSerializableFactory(MatMulSerialization.FACTORY_ID,
                new MatMulSerialization());
        return config;
    }
}
