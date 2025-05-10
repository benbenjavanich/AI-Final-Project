# AI CNA Scheduler

A proof-of-concept discrete-event simulation for AI-driven Certified Nursing Assistant (CNA) call scheduling. This repository implements two dispatch policies (FIFO and urgency-based DynamicScheduler), a tiny PyTorch urgency model, and analysis scripts to compare performance and fairness under varying load and staffing levels.

---

## 🚀 Features

- **SimPy simulation** of patient call arrivals (Poisson process) and CNA dispatch  
- **FIFOScheduler**: classic First-In, First-Out dispatch  
- **DynamicScheduler**: prioritizes calls by an urgency score from a PyTorch neural net  
- **UrgencyModel**: small feed-forward network mapping call type → urgency  
- **Multi-run & sensitivity sweeps** over random seeds, staffing levels, and patient load  
- **Evaluation & plotting** of response-time histograms, workload bar charts, and CSV-exported sensitivity results  

---

## 📁 Repository Structure

AI_CNA_Scheduler/ 
├── data/ 
│ ├── patients.csv # synthetic patient profiles & λ_request 
│ └── cnas.csv # CNA profiles, specialties, shifts 
├── src/ 
│ ├── scheduler.py # FIFOScheduler & DynamicScheduler implementations 
│ ├── simulation.py # main experiment & sensitivity-sweep driver 
│ ├── train_model.py # trains urgency PyTorch model → urgency_model.pt 
│ ├── models.py # UrgencyModel definition & training helper 
│ ├── evaluate.py # plotting helpers & multi-run statistics 
│ └── init.py 
├── sensitivity_results.csv # example output from sensitivity sweep 
├── requirements.txt # Python dependencies 
└── README.md # this file


---

## 💻 Installation

1. **Clone** this repository  
   ```bash
   git clone https://github.com/your-org/AI_CNA_Scheduler.git
   cd AI_CNA_Scheduler

## Create virtual env
python3 -m venv .venv
source .venv/bin/activate

## install dependencies
pip install --upgrade pip
pip install -r requirements.txt


## Usage

1. Train the urgency model
python src/train_model.py --out_path urgency_model.pt
Fits a small neural net on synthetic severity labels and saves urgency_model.pt.

2. Run the simulation & analysis
python src/simulation.py
Generates response-time histograms & workload charts (PNG)
Runs 50-seed multi-run comparison (prints mean ± σ response times)
Executes sensitivity sweeps over staffing (1–3 CNAs) and load multipliers (0.5×, 1×, 1.5×)
Saves full results to sensitivity_results.csv
3. View outputs
Histograms: FIFO*_response_times.png, Dynamic*_response_times.png
Workloads: *_workload.png
Sensitivity CSV: sensitivity_results.csv
🔧 Configuration

Simulation time: Default 8 hours (480 min) in run_simulation(sim_time=8*60)
Service time: Fixed 5 min per call (see scheduler.py)
Urgency heuristic: One-hot call type → severity label → neural net prediction
Adjust these parameters in src/simulation.py and src/scheduler.py as needed.


