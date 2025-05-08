# src/simulation.py

import simpy
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from scheduler import DynamicScheduler, FIFOScheduler
from models import UrgencyModel
from evaluate import plot_response_times, plot_workload

BASE = Path(__file__).parent.parent
DATA = BASE / "data"

def load_data():
    patients = pd.read_csv(
        DATA / "patients.csv",
        converters={"med_times": lambda s: s.split(";")}
    )
    cnas = pd.read_csv(DATA / "cnas.csv")
    return patients, cnas

def patient_generator(env, patient, scheduler):
    """
    Each patient generates calls by a Poisson process, then
    uses the trained PyTorch model to score urgency.
    """
    while True:
        wait = np.random.exponential(1.0 / patient["λ_request"])
        yield env.timeout(wait)

        # one‐hot encode the care_profile
        prof = patient["care_profile"]
        idx  = category_to_index[prof]
        x    = torch.zeros(len(categories))
        x[idx] = 1.0

        # model inference
        with torch.no_grad():
            urgency = float(model(x.unsqueeze(0)))

        scheduler.request_care(env.now, patient["patient_id"], urgency)

def run_simulation(patients_df, cnas_df, sim_time=8*60, sched_cls=DynamicScheduler):
    """
    Runs a single sim for sim_time minutes and returns the scheduler's metrics.
    """
    env   = simpy.Environment()
    sched = sched_cls(env, patients_df, cnas_df)

    # start all patient processes
    for _, row in patients_df.iterrows():
        env.process(patient_generator(env, row, sched))

    # start the scheduler
    env.process(sched.monitor())

    # run to completion
    env.run(until=sim_time)
    return sched.metrics

def multi_run_comparison(patients_df, cnas_df, n_runs=50, sim_time=8*60):
    """
    Runs FIFO vs. Dynamic over n_runs different seeds and
    returns arrays of average response times.
    """
    fifo_avgs = []
    dyn_avgs  = []

    for seed in range(n_runs):
        np.random.seed(seed)

        # deep copy to avoid λ_request contamination
        pts = patients_df.copy()
        cn  = cnas_df.copy()

        m1 = run_simulation(pts, cn, sim_time, FIFOScheduler)
        m2 = run_simulation(pts, cn, sim_time, DynamicScheduler)

        fifo_avgs.append(np.mean(m1["response_times"]))
        dyn_avgs .append(np.mean(m2["response_times"]))

    return np.array(fifo_avgs), np.array(dyn_avgs)

def sensitivity_sweep(patients_df, cnas_df,
                      staff_levels=(1,2,3),
                      intensities=(0.5,1.0,1.5),
                      sim_time=8*60):
    """
    Sweeps over number of CNAs (taking first N rows of cnas_df)
    and call-intensity multipliers, recording avg RT and fairness.
    """
    rows = []
    for staff in staff_levels:
        cn_sub = cnas_df.head(staff).reset_index(drop=True)
        for intensity in intensities:
            # scale call rates
            pts = patients_df.copy()
            pts["λ_request"] = pts["λ_request"] * intensity

            # FIFO vs. Dynamic sim
            m_dyn  = run_simulation(pts, cn_sub, sim_time, DynamicScheduler)
            m_fifo = run_simulation(pts, cn_sub, sim_time, FIFOScheduler)

            # compute metrics
            rt_dyn  = np.mean(m_dyn["response_times"])
            rt_fifo = np.mean(m_fifo["response_times"])

            # fairness = stddev of visits per CNA
            df_dyn   = pd.DataFrame(m_dyn["assignments"], columns=["pt","cna","time"])
            df_fifo  = pd.DataFrame(m_fifo["assignments"], columns=["pt","cna","time"])
            f_dyn    = df_dyn["cna"].value_counts().std()
            f_fifo   = df_fifo["cna"].value_counts().std()

            rows.append({
                "scheduler":     "DynamicScheduler",
                "staff":         staff,
                "intensity":     intensity,
                "avg_rt":        rt_dyn,
                "fairness":      f_dyn
            })
            rows.append({
                "scheduler":     "FIFOScheduler",
                "staff":         staff,
                "intensity":     intensity,
                "avg_rt":        rt_fifo,
                "fairness":      f_fifo
            })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    # 1) Load data
    patients_df, cnas_df = load_data()

    # 2) Load trained PyTorch model
    ckpt = torch.load(BASE/"urgency_model.pt", map_location="cpu")
    categories = ckpt["categories"]
    category_to_index = {c:i for i,c in enumerate(categories)}

    model = UrgencyModel(input_size=len(categories))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 3) Single-run comparison & plots
    for SchedulerClass in (FIFOScheduler, DynamicScheduler):
        metrics = run_simulation(patients_df, cnas_df, sched_cls=SchedulerClass)
        avg_rt  = np.mean(metrics["response_times"])
        print(f"{SchedulerClass.__name__}: Avg RT = {avg_rt:.2f} min")
        plot_response_times(metrics["response_times"], SchedulerClass.__name__)
        plot_workload(metrics["assignments"], SchedulerClass.__name__)

    # 4) Multi-run statistics
    fifo_avgs, dyn_avgs = multi_run_comparison(patients_df, cnas_df)
    print("\n=== Multi-run comparison (50 seeds) ===")
    print(f"FIFO   avg RT = {fifo_avgs.mean():.2f} ± {fifo_avgs.std():.2f} min")
    print(f"Dynamic avg RT = {dyn_avgs.mean():.2f} ± {dyn_avgs.std():.2f} min")

    # 5) Sensitivity sweeps
    df_sens = sensitivity_sweep(patients_df, cnas_df)
    print("\n=== Sensitivity Sweep: Avg RT ===")
    print(df_sens.pivot_table(index=["staff","intensity"],
                              columns="scheduler",
                              values="avg_rt"))
    print("\n=== Sensitivity Sweep: Fairness ===")
    print(df_sens.pivot_table(index=["staff","intensity"],
                              columns="scheduler",
                              values="fairness"))

    # save to CSV for your report
    df_sens.to_csv(BASE/"sensitivity_results.csv", index=False)
    print("\nSaved full sweep results to sensitivity_results.csv")
