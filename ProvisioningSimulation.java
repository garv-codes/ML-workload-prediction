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
 * Drives CloudSim with the workload forecasts produced by the Python MTTD framework.
 *
 * For every time step the number of VMs is chosen from the MTTD forecast (with a small headroom).
 * The real requests of that time step are then executed in CloudSim as cloudlets.
 * If they are not finished within one time step (SLA_SECONDS) the step counts as an SLA violation.
 *
 * Input : predictions_retrain.csv  (columns: hour, actual, predicted, active_model, retrain_event)
 * Output: summary on the console + cloudsim_results.csv (one row per time step)
 */
public class ProvisioningSimulation {

    // ---------- file locations (change if needed) ----------
    // Input / output files. Optional command-line arguments override them:
    //   args[0] = predictions CSV produced by mttd_bridge.py   (default: predictions_retrain.csv)
    //   args[1] = per-step output CSV                          (default: cloudsim_results.csv)
    static String RETRAIN_CSV   = "predictions_retrain.csv";
    static String RESULT_CSV    = "cloudsim_results.csv";

    // ---------- capacity model ----------
    // Length of ONE time step of your dataset in seconds: 3600 = hourly, 300 = 5-minute, 60 = per-minute
    static final double INTERVAL_SECONDS = 3600;
    // Capacity is auto-calibrated so that the AVERAGE load needs about this many VMs
    static final double TARGET_AVG_VMS   = 6;
    // Requests one VM serves per time step. 0 = auto (mean of actual / TARGET_AVG_VMS); or type your own number
    static double HITS_PER_VM_PER_HOUR   = 4000;
    static final double HEADROOM         = 1.10;   // provision 10% extra than the estimate
    static final int    MAX_VMS          = 150;    // datacenter limit
    static final double SLA_SECONDS      = INTERVAL_SECONDS;   // all work must finish within one time step
    static final int    VM_MIPS          = 1000;
    // set in main() once the capacity is known
    static double HITS_PER_CLOUDLET      = 0;      // = capacity / 20
    static double MI_PER_HIT             = 0;      // so that one VM does exactly 'capacity' requests per time step

    // 0 = simulate every hour in the CSV; set e.g. 200 for a quick test run
    static final int MAX_HOURS = 0;

    // ================================================================
    public static void main(String[] args) throws Exception {
        Log.disable(); // CloudSim is very chatty; we print our own summary
        if (args.length > 0) RETRAIN_CSV = args[0];
        if (args.length > 1) RESULT_CSV = args[1];

        double[][] rt = readCsv(RETRAIN_CSV);    // [i][0]=actual  [i][1]=MTTD forecast

        // ---- calibrate the capacity model to this dataset ----
        if (HITS_PER_VM_PER_HOUR <= 0) {
            double sum = 0;
            for (double[] r : rt) sum += r[0];
            HITS_PER_VM_PER_HOUR = Math.max(1, Math.round((sum / rt.length) / TARGET_AVG_VMS));
        }
        HITS_PER_CLOUDLET = HITS_PER_VM_PER_HOUR / 20.0;
        MI_PER_HIT = (VM_MIPS * INTERVAL_SECONDS) / HITS_PER_VM_PER_HOUR;
        System.out.println("Capacity per VM per time step = " + HITS_PER_VM_PER_HOUR
                + " requests, time step = " + INTERVAL_SECONDS + " s, max VMs = " + MAX_VMS + "\n");

        printModelUsage(RETRAIN_CSV);

        int n = rt.length;
        if (MAX_HOURS > 0 && MAX_HOURS < n) n = MAX_HOURS;

        double vmHours = 0, needed = 0, waste = 0, sumMakespan = 0;
        int violations = 0, underHours = 0;

        PrintWriter out = new PrintWriter(RESULT_CSV);
        out.println("hour,actual,forecast,vms_provisioned,vms_needed,completion_s,sla_violated");

        System.out.println("Simulating " + n + " time steps driven by the MTTD forecasts ...");
        long t0 = System.currentTimeMillis();

        for (int h = 0; h < n; h++) {
            double actual = rt[h][0];
            double forecast = rt[h][1];
            int required = vmsFor(actual, 1.0);        // VMs needed if the load were known exactly
            int vms = vmsFor(forecast, HEADROOM);      // VMs provisioned from the MTTD forecast
            double makespan = runOneHour(vms, actual); // real CloudSim run of this time step
            boolean viol = makespan > SLA_SECONDS;

            vmHours += vms;
            needed += required;
            waste += Math.max(0, vms - required);
            sumMakespan += Math.min(makespan, 20000);
            if (viol) violations++;
            if (vms < required) underHours++;

            out.println(h + "," + (long) actual + "," + (long) forecast + "," + vms + ","
                    + required + "," + String.format("%.1f", makespan) + "," + (viol ? 1 : 0));
            if ((h + 1) % 100 == 0)
                System.out.println("  ... " + (h + 1) + "/" + n + " time steps done");
        }
        out.close();

        System.out.println("\nFinished in " + (System.currentTimeMillis() - t0) / 1000 + " s\n");
        System.out.println("=== CloudSim provisioning driven by the MTTD forecasts ===");
        System.out.println(String.format("Time steps simulated            : %d", n));
        System.out.println(String.format("VM-hours provisioned            : %.0f", vmHours));
        System.out.println(String.format("VM-hours needed (exact load)    : %.0f", needed));
        System.out.println(String.format("Wasted VM-hours (over-provision): %.0f", waste));
        System.out.println(String.format("Under-provisioned time steps    : %d (%.1f%%)", underHours, 100.0 * underHours / n));
        System.out.println(String.format("SLA-violating time steps        : %d (%.1f%%)", violations, 100.0 * violations / n));
        System.out.println(String.format("Average completion time (s)     : %.0f", sumMakespan / n));
        System.out.println("\nPer-step details written to " + RESULT_CSV);
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
        return last - first; // time spent executing, ignoring CloudSim's small start offset
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