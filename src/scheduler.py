# src/scheduler.py

import heapq
import numpy as np
import pandas as pd

class DynamicScheduler:
    def __init__(self, env, patients_df: pd.DataFrame, cnas_df: pd.DataFrame):
        """
        env:         simpy.Environment
        patients_df: DataFrame with ['patient_id','care_profile',...]
        cnas_df:     DataFrame with ['cna_id','specialties','shift_start','shift_end',...]
        """
        self.env = env

        # 1) Build patient_id -> care_profile lookup
        self.patient_profiles = (
            patients_df.set_index("patient_id")["care_profile"]
                       .to_dict()
        )

        # 2) Copy & sanitize CNA DataFrame
        cnas = cnas_df.copy()

        # Ensure shift columns are strings, then parse "HH:MM" -> total minutes
        cnas["shift_start"] = (
            cnas["shift_start"].astype(str)
                .apply(lambda t: int(t.split(":")[0]) * 60 + int(t.split(":")[1]))
        )
        cnas["shift_end"] = (
            cnas["shift_end"].astype(str)
                .apply(lambda t: int(t.split(":")[0]) * 60 + int(t.split(":")[1]))
        )
        self.day_start = min(cnas["shift_start"])

        # Split specialties into Python lists
        cnas["specialties"] = (
            cnas["specialties"].astype(str) .apply(lambda s: s.split(";"))
        )

        # 3) Initialize availability & profile dict
        self.available = set(cnas["cna_id"])
        # profiles: cna_id -> { specialties: [...], shift_start: int, shift_end: int, ... }
        self.profiles = cnas.set_index("cna_id").to_dict("index")

        # 4) Prep request queue and metrics
        self.queue = []  # heap of (–urgency, request_ts, patient_id)
        self.metrics = {
            "response_times": [],   # assign_time – request_time
            "assignments": []       # (patient_id, cna_id, assign_time)
        }

    def request_care(self, ts: float, patient_id: str, urgency: float):
        """Called by patient_generator when a call arrives."""
        heapq.heappush(self.queue, (-urgency, ts, patient_id))

    def monitor(self):
        """SimPy process: assign highest‐urgency waiting calls to in‐shift CNAs."""
        while True:
            if self.queue and self.available:
                neg_u, ts_req, pid = heapq.heappop(self.queue)
                urgency = -neg_u
                prof = self.patient_profiles[pid]

                # 1) Perfect matches: skill + in‐shift
                candidates = [
                    c for c in self.available
                    if (prof in self.profiles[c]["specialties"] and self._in_shift(c))
                ]

                # 2) If none, any in‐shift CNA
                if not candidates:
                    candidates = [c for c in self.available if self._in_shift(c)]

                # 3) If still none, requeue and wait 1 minute
                if not candidates:
                    heapq.heappush(self.queue, (neg_u, ts_req, pid))
                    yield self.env.timeout(1)
                    continue

                # Assign to the first candidate
                cna_id = candidates[0]
                self.available.remove(cna_id)

                # Record metrics
                self.metrics["response_times"].append(self.env.now - ts_req)
                self.metrics["assignments"].append((pid, cna_id, self.env.now))

                # Simulate variable care time (Normal(5,2) clipped ≥1)
                duration = max(1, int(np.random.normal(5, 2)))
                self.env.process(self._complete(cna_id, duration))

            # Try again in 1 minute
            yield self.env.timeout(1)

    def _in_shift(self, cna_id: str) -> bool:
        start = self.profiles[cna_id]["shift_start"]
        end   = self.profiles[cna_id]["shift_end"]
        # treat env.now=0 as self.day_start minutes past midnight
        tod   = (self.day_start + int(self.env.now)) % (24 * 60)
        return start <= tod < end
    
    def _complete(self, cna_id: str, duration: float):
        """
        SimPy process that runs while a CNA is busy,
        then marks them available again when done.
        """
        # wait for the care to finish
        yield self.env.timeout(duration)
        # mark that CNA back as available
        self.available.add(cna_id)
        
# in src/scheduler.py

class FIFOScheduler(DynamicScheduler):
    def monitor(self):
        while True:
            if self.queue and self.available:
                # ignore urgency: pop oldest by timestamp
                # since heap is (-urgency, ts, pid), you can peek all, find the min ts
                oldest = min(self.queue, key=lambda x: x[1])
                self.queue.remove(oldest)
                neg_u, ts_req, pid = oldest
                cna = self.available.pop()
                self.metrics["response_times"].append(self.env.now - ts_req)
                self.metrics["assignments"].append((pid, cna, self.env.now))
                self.env.process(self._complete(cna, 5))
            yield self.env.timeout(1)

