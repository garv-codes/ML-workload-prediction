package mttd;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.CloudletSchedulerTimeShared;
import org.cloudbus.cloudsim.Datacenter;
import org.cloudbus.cloudsim.DatacenterBroker;
import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.UtilizationModel;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.VmAllocationPolicySimple;
import org.cloudbus.cloudsim.VmSchedulerTimeShared;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;

/**
 * Drives CloudSim with workload predictions produced by the Python MTTD framework.
 *
 * For every simulated hour we decide how many VMs to provision using 4 policies:
 *   1. REACTIVE            : uses the PREVIOUS hour's real load (what a threshold autoscaler sees)
 *   2. MTTD (no retrain)   : uses predictions_noretrain.csv
 *   3. MTTD (with retrain) : uses predictions_retrain.csv   (full MTTD: 2-strike retraining)
 *   4. ORACLE              : uses the real load of this hour (best possible)
 *   5. MTTD-Hybrid         : max(MTTD forecast, previous hour real load) -> guards against spikes
 *
 * Then a real CloudSim simulation runs that hour's requests on those VMs.
 * The hour's load is turned into cloudlets (1 cloudlet = HITS_PER_CLOUDLET requests).
 * If the cloudlets are not finished within one hour (3600 s of simulated time)
 * the hour counts as an SLA violation.
 */
public class ProvisioningSimulation {

    // ---------- file locations (change if needed) ----------
    static final String RETRAIN_CSV   = "predictions_retrain.csv";
    static final String NORETRAIN_CSV = "predictions_noretrain.csv";
    static final String RESULT_CSV    = "cloudsim_results.csv";

    // ---------- capacity model ----------
    static final double HITS_PER_VM_PER_HOUR = 4000;   // one VM can serve 4000 requests / hour
    static final double HEADROOM             = 1.10;   // provision 10% extra than the estimate
    static final int    MAX_VMS              = 150;    // datacenter limit
    static final double SLA_SECONDS          = 3600;   // all work must finish within the hour
    static final double HITS_PER_CLOUDLET    = 200;

    static final int    VM_MIPS = 1000;
    // MI needed per request, so that one VM (1000 MIPS) does exactly 4000 requests in 3600 s
    static final double MI_PER_HIT = (VM_MIPS * 3600.0) / HITS_PER_VM_PER_HOUR; // = 900 MI

    // 0 = simulate every hour in the CSV; set e.g. 200 for a quick test run
    static final int MAX_HOURS = 100;

    // ================================================================
    public static void main(String[] args) throws Exception {
        Log.disable(); // CloudSim is very chatty; we print our own summary

        double[][] rt = readCsv(RETRAIN_CSV);    // [i][0]=actual [i][1]=predicted
        double[][] nr = readCsv(NORETRAIN_CSV);
        printModelUsage(RETRAIN_CSV);
        printModelUsage(NORETRAIN_CSV);
        int n = rt.length;
        if (MAX_HOURS > 0 && MAX_HOURS < n) n = MAX_HOURS;

        String[] names = { "REACTIVE", "MTTD-noRetrain", "MTTD-Retrain", "ORACLE", "MTTD-Hybrid" };
        double[] vmHours = new double[5], waste = new double[5], sumMakespan = new double[5];
        int[] violations = new int[5], underHours = new int[5];

        PrintWriter out = new PrintWriter(RESULT_CSV);
        out.println("hour,actual,policy,vms,required_vms,makespan_s,sla_violated");

        System.out.println("Simulating " + n + " hours x 5 policies ...");
        long t0 = System.currentTimeMillis();

        for (int h = 0; h < n; h++) {
            double actual = rt[h][0];
            int required = vmsFor(actual, 1.0);

            double[] estimate = new double[5];
            estimate[0] = (h == 0) ? actual : rt[h - 1][0]; // reactive: last hour's real load
            estimate[1] = nr[h][1];
            estimate[2] = rt[h][1];
            estimate[3] = actual;
            estimate[4] = Math.max(rt[h][1], estimate[0]); // hybrid: max(MTTD forecast, last real load)

            for (int p = 0; p < 5; p++) {
                int vms = vmsFor(estimate[p], HEADROOM);
                double makespan = runOneHour(vms, actual);
                boolean viol = makespan > SLA_SECONDS;

                vmHours[p] += vms;
                waste[p] += Math.max(0, vms - required);
                sumMakespan[p] += Math.min(makespan, 20000);
                if (viol) violations[p]++;
                if (vms < required) underHours[p]++;

                out.println(h + "," + (long) actual + "," + names[p] + "," + vms + ","
                        + required + "," + String.format("%.1f", makespan) + "," + (viol ? 1 : 0));
            }
            if ((h + 1) % 100 == 0)
                System.out.println("  ... " + (h + 1) + "/" + n + " hours done");
        }
        out.close();

        System.out.println("\nFinished in " + (System.currentTimeMillis() - t0) / 1000 + " s\n");
        System.out.println(String.format("%-16s %10s %12s %12s %14s %12s",
                "Policy", "VM-hours", "Wasted VMh", "SLA viol.", "Under-prov hrs", "Avg resp (s)"));
        for (int p = 0; p < 5; p++) {
            System.out.println(String.format("%-16s %10.0f %12.0f %7d (%4.1f%%) %9d %14.0f",
                    names[p], vmHours[p], waste[p], violations[p], 100.0 * violations[p] / n,
                    underHours[p], sumMakespan[p] / n));
        }
        System.out.println("\nPer-hour details written to " + RESULT_CSV);
    }

    // ================================================================
    /** Number of VMs needed for an estimated load of 'hits' requests in the next hour. */
    static int vmsFor(double hits, double headroom) {
        int v = (int) Math.ceil(hits * headroom / HITS_PER_VM_PER_HOUR);
        return Math.max(1, Math.min(MAX_VMS, v));
    }

    /**
     * Runs ONE full CloudSim simulation: 'vms' VMs, 'hits' requests turned into cloudlets.
     * Returns the time (simulated seconds) at which the last cloudlet finished.
     */
    static double runOneHour(int vms, double hits) throws Exception {
        CloudSim.init(1, Calendar.getInstance(), false);

        createDatacenter("Datacenter_0");
        DatacenterBroker broker = new DatacenterBroker("Broker_0");
        int brokerId = broker.getId();

        // ---- VMs ----
        List<Vm> vmList = new ArrayList<Vm>();
        for (int i = 0; i < vms; i++) {
            vmList.add(new Vm(i, brokerId, VM_MIPS, 1, 512, 1000, 10000, "Xen",
                    new CloudletSchedulerTimeShared()));
        }
        broker.submitVmList(vmList);

        // ---- Cloudlets (the requests of this hour) ----
        int numCloudlets = (int) Math.ceil(hits / HITS_PER_CLOUDLET);
        long length = (long) (HITS_PER_CLOUDLET * MI_PER_HIT);
        UtilizationModel full = new UtilizationModelFull();
        List<Cloudlet> cloudletList = new ArrayList<Cloudlet>();
        for (int i = 0; i < numCloudlets; i++) {
            Cloudlet c = new Cloudlet(i, length, 1, 300, 300, full, full, full);
            c.setUserId(brokerId);
            cloudletList.add(c);
        }
        broker.submitCloudletList(cloudletList);

        CloudSim.startSimulation();
        List<Cloudlet> finished = broker.getCloudletReceivedList();
        CloudSim.stopSimulation();

        if (finished.size() < numCloudlets) return 1e9; // some cloudlets never finished

        double first = Double.MAX_VALUE, last = 0;
        for (Cloudlet c : finished) {
            first = Math.min(first, c.getExecStartTime());
            last = Math.max(last, c.getFinishTime());
        }
        return last - first; // time actually spent working, ignoring CloudSim's 0.2 s start offset
    }

    /** 25 hosts x 8 cores x 1000 MIPS = room for 200 single-core VMs. */
    static Datacenter createDatacenter(String name) throws Exception {
        List<Host> hostList = new ArrayList<Host>();
        for (int h = 0; h < 25; h++) {
            List<Pe> peList = new ArrayList<Pe>();
            for (int c = 0; c < 8; c++) peList.add(new Pe(c, new PeProvisionerSimple(VM_MIPS)));
            hostList.add(new Host(h, new RamProvisionerSimple(16384),
                    new BwProvisionerSimple(1000000), 1000000, peList,
                    new VmSchedulerTimeShared(peList)));
        }
        DatacenterCharacteristics ch = new DatacenterCharacteristics(
                "x86", "Linux", "Xen", hostList, 10.0, 3.0, 0.05, 0.001, 0.0);
        return new Datacenter(name, ch, new VmAllocationPolicySimple(hostList),
                new LinkedList<Storage>(), 0);
    }
    /** Prints which model MTTD used (column active_model) and when it switched. */
    static void printModelUsage(String file) throws Exception {
        java.util.Map<String, Integer> hoursPerModel = new java.util.LinkedHashMap<String, Integer>();
        List<String> switches = new ArrayList<String>();
        BufferedReader br = new BufferedReader(new FileReader(file));
        br.readLine(); // header: hour,actual,predicted,active_model,retrain_event
        String line, prevModel = null;
        int total = 0;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] p = line.split(",");
            int hour = (int) Double.parseDouble(p[0].trim());
            String model = p[3].trim();
            Integer old = hoursPerModel.get(model);
            hoursPerModel.put(model, old == null ? 1 : old + 1);
            if (prevModel != null && !model.equals(prevModel))
                switches.add("hour " + hour + ": " + prevModel + " -> " + model);
            prevModel = model;
            total++;
        }
        br.close();

        System.out.println("=== Models used by MTTD (" + file + ") ===");
        for (java.util.Map.Entry<String, Integer> e : hoursPerModel.entrySet())
            System.out.println(String.format("  %-16s %5d hours (%4.1f%%)",
                    e.getKey(), e.getValue(), 100.0 * e.getValue() / total));
        if (switches.isEmpty()) System.out.println("  No model switches");
        for (String s : switches) System.out.println("  switch at " + s);
        System.out.println();
    }

    /** Reads predictions CSV -> rows of {actual, predicted}. Header: hour,actual,predicted,... */
    static double[][] readCsv(String file) throws Exception {
        List<double[]> rows = new ArrayList<double[]>();
        BufferedReader br = new BufferedReader(new FileReader(file));
        br.readLine(); // header
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] p = line.split(",");
            rows.add(new double[] { Double.parseDouble(p[1]), Double.parseDouble(p[2]) });
        }
        br.close();
        return rows.toArray(new double[0][]);
    }
}

