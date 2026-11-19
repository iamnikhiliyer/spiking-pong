"""A fixed Izhikevich population and a trainable actor–critic for Pong.

Coordinates and errors use a unit-height court. Time in the neuron equations
is in milliseconds. The controller samples an action every 50 milliseconds.
"""

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np


class IzhikevichPopulation:
    """Regular-spiking E and fast-spiking I cells; time in ms, current in model units.

    Equations: Izhikevich 2003, https://www.izhikevich.org/publications/spikes.pdf
    Both state variables use two half Euler steps, with threshold/reset each half.
    """

    def __init__(self, size, n_excitatory, dt_ms=1.0):
        self.dt_ms = dt_ms
        self.a = np.full(size, 0.1)
        self.a[:n_excitatory] = 0.02
        self.b = 0.2
        self.c = -65.0
        self.d = np.full(size, 2.0)
        self.d[:n_excitatory] = 8.0
        self.v = np.full(size, -65.0)
        self.u = self.b * self.v.copy()

    def step(self, current):
        fired = np.zeros(len(self.v), dtype=bool)
        half = self.dt_ms / 2
        for _ in range(2):
            self.v += half * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + current)
            self.u += half * self.a * (self.b * self.v - self.u)
            crossed = self.v >= 30
            fired |= crossed
            self.v[crossed] = self.c
            self.u[crossed] += self.d[crossed]
        if not np.isfinite(self.v).all() or not np.isfinite(self.u).all():
            raise FloatingPointError(
                "Neuron state diverged; inspect the numerical configuration"
            )
        return fired


@dataclass(frozen=True)
class AssemblyConfig:
    assemblies: int = 9
    e_per_assembly: int = 8
    i_per_assembly: int = 2
    input_current: float = 14.0
    tuning_width: float = 0.12
    noise_sd: float = 2.0
    recurrent_gain: float = 1.0
    feature_tau_ms: float = 40.0

    def validate(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("assemblies", "e_per_assembly", "i_per_assembly"):
            if not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an integer")
        if self.assemblies < 3:
            raise ValueError("At least three assemblies are required")
        return self

    @property
    def n_excitatory(self):
        return self.assemblies * self.e_per_assembly

    @property
    def n_neurons(self):
        return self.n_excitatory + self.assemblies * self.i_per_assembly


def make_connectivity(config, seed, topology="structured"):
    """Return a signed recurrent matrix, indexed by postsynaptic then presynaptic cell."""
    if topology not in ("structured", "shuffled", "none"):
        raise ValueError(f"Unknown topology: {topology}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 10]))
    n, ne = config.n_neurons, config.n_excitatory
    groups = np.r_[
        np.repeat(np.arange(config.assemblies), config.e_per_assembly),
        np.repeat(np.arange(config.assemblies), config.i_per_assembly),
    ]
    local = groups[:, None] == groups[None, :]
    probability = np.where(local, 0.35, 0.015)
    probability[ne:, :ne] = np.where(local[ne:, :ne], 0.6, 0.06)
    probability[:, ne:] = 0.25
    weights = rng.uniform(1, 2, (n, n))
    weights[:, ne:] = -rng.uniform(1, 4, (n, n - ne))
    weights *= rng.random((n, n)) < probability
    np.fill_diagonal(weights, 0)
    if topology == "shuffled":
        # Preserve every postsynaptic cell's incoming E/I weights and degree.
        for post in range(n):
            for sources in (np.arange(ne), np.arange(ne, n)):
                sources = sources[sources != post]
                weights[post, sources] = rng.permutation(weights[post, sources])
    if topology == "none":
        weights.fill(0)
    return weights * config.recurrent_gain


class SpikingAssemblies:
    """Encode the vertical gap as current and pool excitatory spikes into nine rates."""

    def __init__(self, config, seed, topology="structured", noise_seed=None):
        self.config = config.validate()
        self.weights = make_connectivity(config, seed, topology)
        self.rng = np.random.default_rng(
            np.random.SeedSequence([seed if noise_seed is None else noise_seed, 11])
        )
        self.cells = IzhikevichPopulation(config.n_neurons, config.n_excitatory)
        self.synaptic_current = np.zeros(config.n_neurons)
        self.previous_spikes = np.zeros(config.n_neurons, dtype=bool)
        self.filtered_activity = np.zeros(config.assemblies)
        self.centers = np.linspace(-0.8, 0.8, config.assemblies)
        self.synaptic_decay = np.exp(-1 / 5)
        self.feature_decay = np.exp(-1 / config.feature_tau_ms)

    def step(self, relative_position):
        c = self.config
        self.synaptic_current *= self.synaptic_decay
        if self.previous_spikes.any():
            self.synaptic_current += self.weights[:, self.previous_spikes].sum(axis=1)
        tuning = np.exp(
            -0.5 * ((relative_position - self.centers) / c.tuning_width) ** 2
        )
        current = (
            2 + self.rng.normal(0, c.noise_sd, c.n_neurons) + self.synaptic_current
        )
        current[: c.n_excitatory] += np.repeat(
            c.input_current * tuning, c.e_per_assembly
        )
        fired = self.cells.step(current)
        self.previous_spikes = fired
        self.filtered_activity *= self.feature_decay
        self.filtered_activity += (
            fired[: c.n_excitatory].reshape(c.assemblies, c.e_per_assembly).mean(axis=1)
        )
        return fired

    def features(self):
        # The actor sees filtered spikes plus a constant bias, no game coordinates.
        rates = self.filtered_activity * (1000 * (1 - self.feature_decay))
        return np.r_[np.clip(rates / 40, 0, 3), 1.0]


class PopulationActorCritic:
    """Linear stochastic policy and value estimator, trained by reward prediction error."""

    def __init__(
        self,
        n_features,
        actor_lr=0.04,
        critic_lr=0.05,
        gamma=0.98,
        trace_lambda=0.8,
        entropy=0.001,
    ):
        self.weights = np.zeros((3, n_features))
        self.critic = np.zeros(n_features)
        self.actor_trace = np.zeros_like(self.weights)
        self.critic_trace = np.zeros(n_features)
        self.actor_lr, self.critic_lr = actor_lr, critic_lr
        self.gamma, self.trace_lambda, self.entropy = gamma, trace_lambda, entropy

    def probabilities(self, spike_features):
        logits = self.weights @ spike_features
        probabilities = np.exp(logits - logits.max())
        return probabilities / probabilities.sum()

    def reset_traces(self):
        self.actor_trace.fill(0)
        self.critic_trace.fill(0)

    def learn(self, features, action, reward, next_features, terminal, enabled=True):
        if not enabled:
            return 0.0
        probabilities = self.probabilities(features)
        selected = np.eye(3)[action]
        gradient = np.outer(selected - probabilities, features)
        self.actor_trace *= self.gamma * self.trace_lambda
        self.actor_trace += gradient
        self.critic_trace *= self.gamma * self.trace_lambda
        self.critic_trace += features
        value = float(self.critic @ features)
        next_value = 0.0 if terminal else float(self.critic @ next_features)
        delta = float(np.clip(reward + self.gamma * next_value - value, -2, 2))
        entropy = -float(probabilities @ np.log(probabilities + 1e-12))
        entropy_gradient = -np.outer(
            probabilities * (np.log(probabilities + 1e-12) + entropy), features
        )
        self.weights += self.actor_lr * (
            delta * self.actor_trace + self.entropy * entropy_gradient
        )
        self.critic += self.critic_lr * delta * self.critic_trace
        np.clip(self.weights, -12, 12, out=self.weights)
        np.clip(self.critic, -20, 20, out=self.critic)
        if terminal:
            self.reset_traces()
        return delta


class Pong:
    """Unit-square court with a single paddle on the right. A serve ends at its first encounter."""

    paddle_x = 0.94
    paddle_half_height = 0.11
    ball_radius = 0.012
    paddle_speed = 0.8

    def __init__(self, rng):
        self.rng = rng
        self.paddle = 0.5
        self.restart_rally()

    def restart_rally(self):
        self.x = 0.08
        self.y = float(self.rng.uniform(0.1, 0.9))
        self.vx = 0.55
        self.vy = float(self.rng.choice([-1, 1]) * self.rng.uniform(0.2, 0.45))

    def step(self, dt_s, action):
        self.paddle = float(
            np.clip(
                self.paddle + action * self.paddle_speed * dt_s,
                self.paddle_half_height,
                1 - self.paddle_half_height,
            )
        )
        self.x += self.vx * dt_s
        self.y += self.vy * dt_s
        if self.y < self.ball_radius:
            self.y = 2 * self.ball_radius - self.y
            self.vy = abs(self.vy)
        elif self.y > 1 - self.ball_radius:
            self.y = 2 * (1 - self.ball_radius) - self.y
            self.vy = -abs(self.vy)
        if self.x < self.ball_radius:
            self.x = 2 * self.ball_radius - self.x
            self.vx = abs(self.vx)
        if self.x >= self.paddle_x and self.vx > 0:
            self.x = 2 * self.paddle_x - self.x
            self.vx = -abs(self.vx)
            return (
                "hit"
                if abs(self.y - self.paddle)
                <= self.paddle_half_height + self.ball_radius
                else "miss"
            )
        return None


@dataclass(frozen=True)
class LearningConfig:
    population: AssemblyConfig = field(default_factory=AssemblyConfig)
    control_ms: int = 50
    train_trials: int = 320
    eval_trials: int = 100
    actor_lr: float = 0.04
    critic_lr: float = 0.05
    gamma: float = 0.98
    trace_lambda: float = 0.8
    entropy: float = 0.001
    shaping_gain: float = 2.0
    replay_seconds: float = 20.0

    def validate(self):
        self.population.validate()
        for name, value in vars(self).items():
            if name != "population" and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("control_ms", "train_trials", "eval_trials"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.gamma <= 1 or not 0 <= self.trace_lambda <= 1:
            raise ValueError("Invalid discount or eligibility decay")
        if self.actor_lr <= 0 or self.critic_lr <= 0 or self.replay_seconds <= 0:
            raise ValueError("Learning rates and replay length must be positive")
        return self

    def actor(self):
        self.validate()
        return PopulationActorCritic(
            self.population.assemblies + 1,
            self.actor_lr,
            self.critic_lr,
            self.gamma,
            self.trace_lambda,
            self.entropy,
        )


def shaped_reward(old_error, new_error, outcome, gamma, gain):
    # Ng, Harada & Russell 1999: F = gamma*Phi(next) - Phi(current).
    # Terminal potential is zero; no teacher action is passed to the learner.
    # https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    terminal_reward = 1.0 if outcome == "hit" else -1.0 if outcome == "miss" else 0.0
    next_potential = 0.0 if outcome else -new_error
    return terminal_reward + gain * (gamma * next_potential + old_error)


def run_phase(
    config,
    seed,
    actor,
    n_trials,
    training=False,
    topology="structured",
    learning=True,
    reward_mode="normal",
    feature_mode="normal",
    record=False,
):
    """Actor receives neural features only; scalar rewards enter learn(), never act()."""
    config.validate()
    if not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    phase = 20 if training else 30
    serve_rng, action_rng, reward_rng = [
        np.random.default_rng(np.random.SeedSequence([seed, phase, k]))
        for k in (1, 2, 3)
    ]
    network = SpikingAssemblies(
        config.population, seed, topology, noise_seed=seed + phase * 10000
    )
    permutation = np.random.default_rng(seed + 9000).permutation(
        config.population.assemblies
    )
    game = Pong(serve_rng)
    steps = 0
    total_spikes = np.zeros(config.population.n_neurons, dtype=int)
    trials, frames, rewards_seen = [], [], []
    actor.reset_traces()
    frame_spikes = np.zeros(config.population.n_neurons, dtype=bool)

    def features():
        values = network.features()
        if feature_mode == "zero_spikes":
            values[:-1] = 0
        elif feature_mode == "scrambled_spikes":
            values[:-1] = values[:-1][permutation]
        elif feature_mode != "normal":
            raise ValueError(f"Unknown feature mode: {feature_mode}")
        return values

    for trial in range(n_trials):
        game.restart_rally()
        game.paddle = float(
            serve_rng.uniform(game.paddle_half_height, 1 - game.paddle_half_height)
        )
        initial_y, initial_paddle = game.y, game.paddle
        outcome, total_reward, trial_steps = None, 0.0, 0
        while outcome is None:
            observed = features()
            probabilities = actor.probabilities(observed)
            action_id = int(action_rng.choice(3, p=probabilities))
            action = (-1.0, 0.0, 1.0)[action_id]
            old_error = abs(game.y - game.paddle)
            for _ in range(config.control_ms):
                fired = network.step(game.y - game.paddle)
                total_spikes += fired
                frame_spikes |= fired
                outcome = game.step(0.001, action)
                steps += 1
                trial_steps += 1
                if record and steps <= config.replay_seconds * 1000 and steps % 50 == 0:
                    frames.append(
                        {
                            "t": steps / 1000,
                            "trial": trial,
                            "ball": [round(game.x, 5), round(game.y, 5)],
                            "paddle": round(game.paddle, 5),
                            "action": action_id,
                            "spikes": np.flatnonzero(frame_spikes).tolist(),
                            "features": np.round(observed, 4).tolist(),
                            "probabilities": np.round(probabilities, 4).tolist(),
                        }
                    )
                if steps % 50 == 0:
                    frame_spikes.fill(False)
                if outcome:
                    break
            next_features = features()
            terminal = outcome is not None
            reward = shaped_reward(
                old_error,
                abs(game.y - game.paddle),
                outcome,
                config.gamma,
                config.shaping_gain,
            )
            total_reward += reward
            if reward_mode == "resampled_reward":
                delivered = (
                    float(reward_rng.choice(rewards_seen)) if rewards_seen else 0.0
                )
                rewards_seen.append(reward)
                if len(rewards_seen) > 512:
                    rewards_seen.pop(0)
            elif reward_mode == "normal":
                delivered = reward
            else:
                raise ValueError(f"Unknown reward mode: {reward_mode}")
            actor.learn(
                observed,
                action_id,
                delivered,
                next_features,
                terminal,
                enabled=training and learning,
            )
        trials.append(
            {
                "trial": trial,
                "success": int(outcome == "hit"),
                "initial_ball_y": initial_y,
                "initial_paddle_y": initial_paddle,
                "final_error": abs(game.y - game.paddle),
                "reward": total_reward,
                "duration_s": trial_steps / 1000,
            }
        )
    return {
        "trials": trials,
        "frames": frames,
        "success_fraction": float(np.mean([row["success"] for row in trials])),
        "mean_firing_rate_hz": float(total_spikes.mean() / (steps / 1000)),
        "max_firing_rate_hz": float(total_spikes.max() / (steps / 1000)),
        "simulated_seconds": steps / 1000,
        "recurrent_weights": network.weights,
    }


def run_learning(
    config, seed, topology="structured", reward_mode="normal", record=False
):
    """Train on independent serves and evaluate frozen weights on matched held-out serves."""
    started = time.perf_counter()
    actor = config.actor()
    initial = actor.weights.copy()
    before = run_phase(
        config, seed, actor, config.eval_trials, topology=topology, record=record
    )
    train = run_phase(
        config,
        seed,
        actor,
        config.train_trials,
        training=True,
        topology=topology,
        reward_mode=reward_mode,
    )
    learned = actor.weights.copy()
    after = run_phase(
        config, seed, actor, config.eval_trials, topology=topology, record=record
    )
    np.testing.assert_array_equal(learned, actor.weights)
    ablations = {
        mode: run_phase(
            config,
            seed,
            actor,
            config.eval_trials,
            topology=topology,
            feature_mode=mode,
        )
        for mode in ("zero_spikes", "scrambled_spikes")
    }
    return {
        "seed": seed,
        "topology": topology,
        "reward_mode": reward_mode,
        "config": asdict(config),
        "before": before,
        "train": train,
        "after": after,
        "ablations": ablations,
        "initial_weights": initial,
        "learned_weights": learned,
        "critic_weights": actor.critic.copy(),
        "elapsed_s": time.perf_counter() - started,
    }
