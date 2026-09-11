"""
Phase 4: Continuous Deployment -- the adaptive retraining threshold logic
from Fig. 3 (steps 14-28) and Eq. 5-6.

Threshold_{t+1} = Score_t + l1 * Score_t          (Eq. 5)
Threshold_{t+2} = Score_{t+1} + l2 * Score_{t+1}  (Eq. 6)

Behaviour (mirrors the flowchart in Fig. 4):
  - Deploy the active model, run it over a deployment cycle (n minutes),
    compute its Score for that cycle.
  - Compare Score against the current threshold.
      - If Score <= threshold: model is healthy. Reset the "strike" counter
        and set a fresh threshold from this cycle's score for next time.
      - If Score > threshold (1st strike): tighten the threshold using l2
        and continue -- give the model one more cycle before retraining.
      - If Score > threshold again on the very next cycle (2nd strike):
        trigger retraining and model reselection.
"""

from dataclasses import dataclass, field
from typing import Optional

from arima_baseline import performance_score


L1_DEFAULT = 0.75   # paper's default weight for first threshold (Eq. 5)
L2_DEFAULT = 0.50   # paper's default weight for second threshold (Eq. 6)


@dataclass
class DeploymentState:
    active_model_name: str
    threshold: float
    strikes: int = 0                 # 0 = healthy, 1 = one strike (watching)
    history: list = field(default_factory=list)  # log of each cycle's result


class DeploymentManager:
    def __init__(self, active_model_name, initial_score, l1=L1_DEFAULT, l2=L2_DEFAULT):
        """initial_score: the Score_t from the Testing phase (Phase 3) that
        got this model selected as active in the first place."""
        self.l1 = l1
        self.l2 = l2
        self.state = DeploymentState(
            active_model_name=active_model_name,
            threshold=initial_score + l1 * initial_score,  # Eq. 5
        )

    def run_cycle(self, actual_values, predicted_values):
        """Feed in one deployment cycle's actual vs predicted values.
        Returns a dict: {'score', 'threshold_used', 'retrain_triggered'}.
        """
        score = performance_score(actual_values, predicted_values)
        threshold_used = self.state.threshold
        retrain_triggered = False

        if score <= threshold_used:
            # Healthy cycle -- reset strikes, refresh threshold from this score
            self.state.strikes = 0
            self.state.threshold = score + self.l1 * score  # Eq. 5, reset
        else:
            self.state.strikes += 1
            if self.state.strikes == 1:
                # First strike: tighten threshold (Eq. 6), give one more chance
                self.state.threshold = score + self.l2 * score
            else:
                # Second consecutive strike: trigger retraining
                retrain_triggered = True
                self.state.strikes = 0  # reset after handling

        result = {
            "active_model": self.state.active_model_name,
            "score": score,
            "threshold_used": threshold_used,
            "strikes": self.state.strikes,
            "retrain_triggered": retrain_triggered,
        }
        self.state.history.append(result)
        return result

    def set_active_model(self, model_name, fresh_score):
        """Called after a retrain + reselect cycle picks a (possibly new)
        active model. Resets threshold state, Eq. 5, using the fresh
        Test-phase score of the newly selected model."""
        self.state.active_model_name = model_name
        self.state.threshold = fresh_score + self.l1 * fresh_score
        self.state.strikes = 0


if __name__ == "__main__":
    import numpy as np
    from synthetic_workload import generate_uniform_workload, generate_random_workload
    from model_selector import evaluate_all_models, select_active_model

    print("=== Simulating deployment with a workload REGIME CHANGE ===")
    print("(uniform workload for the first half, random for the second half")
    print(" -- this should force the framework to detect degradation and")
    print(" trigger retraining, similar to the paper's Fig. 15 comparison)\n")

    # Phase 2/3: initial training + testing on uniform workload
    df_initial = generate_uniform_workload(duration_min=400, seed=1)
    results = evaluate_all_models(df_initial, train_frac=0.75, test_len=80)
    active_model, scores = select_active_model(results)
    print(f"Initial active model (Testing phase): {active_model}  "
          f"(scores: { {k: round(v,2) for k,v in scores.items()} })")

    manager = DeploymentManager(active_model, initial_score=scores[active_model])
    print(f"Initial threshold: {manager.state.threshold:.2f}%\n")

    # Simulate deployment cycles. Cycles 1-2: still uniform-like data (healthy).
    # Cycles 3-4: switch to random-pattern data (should degrade and trigger retrain).
    rng = np.random.default_rng(0)

    def fake_cycle(pattern, n=40, drift=0.0, noise_level=2):
        """Simulates 'actual vs predicted' for a deployment cycle. We don't
        re-run the full model here (that's the sliding-window/ARIMA update
        loop from predictors.py in a live system) -- for this demo we
        approximate degrading accuracy when the pattern shifts, to exercise
        the threshold state machine end to end. Note: because the threshold
        formula (score * (1 + l)) always sits above the score that produced
        it, a single bad cycle never retriggers on its own -- degradation
        must ESCALATE across consecutive cycles to trigger a retrain. This
        matches the paper's stated 'error amplification' design goal.
        """
        actual = np.linspace(100, 100 + drift, n) + rng.normal(0, 2, n)
        predicted = actual + rng.normal(0, noise_level, n)
        return actual, predicted

    cycles = [
        ("healthy", fake_cycle("healthy", noise_level=2)),
        ("healthy", fake_cycle("healthy", noise_level=2)),
        ("degraded", fake_cycle("degraded", noise_level=40)),
        ("worse", fake_cycle("worse", noise_level=90)),  # escalates -> should trigger
    ]

    for i, (label, (actual, predicted)) in enumerate(cycles, start=1):
        result = manager.run_cycle(actual, predicted)
        print(f"Cycle {i} [{label:8s}] -> score={result['score']:.2f}%  "
              f"threshold_used={result['threshold_used']:.2f}%  "
              f"strikes={result['strikes']}  "
              f"retrain_triggered={result['retrain_triggered']}")

        if result["retrain_triggered"]:
            print("   >>> RETRAIN TRIGGERED: retraining all models on fresh data...")
            df_new = generate_random_workload(duration_min=400, seed=2)
            new_results = evaluate_all_models(df_new, train_frac=0.75, test_len=80)
            new_active, new_scores = select_active_model(new_results)
            print(f"   >>> New active model after retrain: {new_active} "
                  f"(scores: { {k: round(v,2) for k,v in new_scores.items()} })")
            manager.set_active_model(new_active, new_scores[new_active])
            print(f"   >>> Threshold reset to {manager.state.threshold:.2f}%")
