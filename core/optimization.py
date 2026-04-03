"""
Parameter Optimization Framework - Bayesian and grid-based optimization.

Features:
- Optuna integration for Bayesian optimization
- Grid search and random search
- Walk-forward optimization
- Sensitivity analysis
- Parameter stability testing
"""

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try to import optuna, fall back gracefully if not installed
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Install with: pip install optuna")


@dataclass
class Parameter:
    """Definition of a parameter to optimize."""
    name: str
    param_type: str  # "float", "int", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List] = None
    step: Optional[float] = None
    log_scale: bool = False
    default: Any = None
    
    def __post_init__(self):
        if self.param_type in ["float", "int"] and (self.low is None or self.high is None):
            raise ValueError(f"Parameter {self.name} needs low and high bounds")
        if self.param_type == "categorical" and not self.choices:
            raise ValueError(f"Categorical parameter {self.name} needs choices")


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    best_params: Dict[str, Any]
    best_value: float
    all_trials: List[Dict]
    optimization_time_seconds: float
    n_trials: int
    objective_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": self.n_trials,
            "objective_name": self.objective_name,
            "optimization_time_seconds": self.optimization_time_seconds,
            "timestamp": self.timestamp.isoformat(),
            "all_trials": self.all_trials
        }


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis."""
    parameter_name: str
    values: List[float]
    objective_values: List[float]
    baseline_value: float
    baseline_objective: float
    sensitivity_score: float  # How much the objective changes per unit param change
    is_stable: bool


class ObjectiveFunction(ABC):
    """Base class for optimization objectives."""
    
    @abstractmethod
    def evaluate(self, params: Dict[str, Any]) -> float:
        """
        Evaluate the objective function with given parameters.
        Returns a scalar value to minimize (or maximize with direction flag).
        """
        pass
    
    @property
    def direction(self) -> str:
        """'minimize' or 'maximize'"""
        return "maximize"
    
    @property
    def name(self) -> str:
        """Name of the objective."""
        return "objective"


class BacktestObjective(ObjectiveFunction):
    """
    Objective function that runs a backtest with given parameters.
    """
    
    def __init__(
        self,
        backtest_runner: Callable,
        metric: str = "sharpe_ratio",
        direction: str = "maximize"
    ):
        self.backtest_runner = backtest_runner
        self.metric = metric
        self._direction = direction
    
    @property
    def direction(self) -> str:
        return self._direction
    
    @property
    def name(self) -> str:
        return f"backtest_{self.metric}"
    
    def evaluate(self, params: Dict[str, Any]) -> float:
        """Run backtest and return the specified metric."""
        try:
            results = self.backtest_runner(params)
            
            if isinstance(results, dict):
                value = results.get(self.metric, 0)
            else:
                value = float(results)
            
            # Handle edge cases
            if np.isnan(value) or np.isinf(value):
                return -1000 if self._direction == "maximize" else 1000
            
            return value
            
        except Exception as e:
            logger.error(f"Backtest failed with params {params}: {e}")
            return -1000 if self._direction == "maximize" else 1000


class ParameterOptimizer:
    """
    Main parameter optimization class supporting multiple methods.
    """
    
    def __init__(
        self,
        parameters: List[Parameter],
        objective: ObjectiveFunction,
        results_dir: str = "results/optimization"
    ):
        self.parameters = {p.name: p for p in parameters}
        self.objective = objective
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self._study: Optional[Any] = None
        self._results: List[OptimizationResult] = []
    
    def optimize_bayesian(
        self,
        n_trials: int = 100,
        timeout_seconds: int = None,
        n_startup_trials: int = 10,
        seed: int = 42,
        show_progress: bool = True
    ) -> OptimizationResult:
        """
        Run Bayesian optimization using Optuna's TPE sampler.
        
        Args:
            n_trials: Number of optimization trials
            timeout_seconds: Maximum time for optimization
            n_startup_trials: Random trials before TPE kicks in
            seed: Random seed for reproducibility
            show_progress: Show progress bar
        
        Returns:
            OptimizationResult with best parameters and history
        """
        if not OPTUNA_AVAILABLE:
            logger.error("Optuna not installed. Using grid search instead.")
            return self.optimize_grid(n_samples=n_trials)
        
        start_time = datetime.now()
        
        # Create Optuna study
        direction = self.objective.direction
        sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
        
        self._study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            study_name=f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Define objective wrapper for Optuna
        def optuna_objective(trial):
            params = self._suggest_params(trial)
            return self.objective.evaluate(params)
        
        # Run optimization
        self._study.optimize(
            optuna_objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=show_progress,
            catch=(Exception,)
        )
        
        # Collect results
        elapsed = (datetime.now() - start_time).total_seconds()
        
        all_trials = [
            {
                "trial_number": t.number,
                "params": t.params,
                "value": t.value,
                "state": str(t.state)
            }
            for t in self._study.trials
        ]
        
        result = OptimizationResult(
            best_params=self._study.best_params,
            best_value=self._study.best_value,
            all_trials=all_trials,
            optimization_time_seconds=elapsed,
            n_trials=len(self._study.trials),
            objective_name=self.objective.name
        )
        
        self._results.append(result)
        self._save_result(result, "bayesian")
        
        logger.info(f"Bayesian optimization complete: {result.best_value:.4f} "
                   f"with params {result.best_params}")
        
        return result
    
    def optimize_grid(
        self,
        n_samples: int = 100,
        grid_points: int = 5
    ) -> OptimizationResult:
        """
        Grid search optimization.
        
        Args:
            n_samples: Max samples if grid is too large
            grid_points: Number of points per parameter
        """
        start_time = datetime.now()
        
        # Generate grid
        grid = self._generate_grid(grid_points)
        
        # Sample if too large
        if len(grid) > n_samples:
            indices = np.random.choice(len(grid), n_samples, replace=False)
            grid = [grid[i] for i in indices]
        
        # Evaluate all points
        results = []
        for params in grid:
            value = self.objective.evaluate(params)
            results.append({"params": params, "value": value})
        
        # Find best
        if self.objective.direction == "maximize":
            best = max(results, key=lambda x: x["value"])
        else:
            best = min(results, key=lambda x: x["value"])
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            best_params=best["params"],
            best_value=best["value"],
            all_trials=[{"trial_number": i, **r} for i, r in enumerate(results)],
            optimization_time_seconds=elapsed,
            n_trials=len(results),
            objective_name=self.objective.name
        )
        
        self._results.append(result)
        self._save_result(result, "grid")
        
        return result
    
    def optimize_random(
        self,
        n_trials: int = 100,
        seed: int = 42
    ) -> OptimizationResult:
        """
        Random search optimization.
        Often competitive with Bayesian for well-behaved objectives.
        """
        np.random.seed(seed)
        start_time = datetime.now()
        
        results = []
        for i in range(n_trials):
            params = self._random_params()
            value = self.objective.evaluate(params)
            results.append({"params": params, "value": value, "trial": i})
        
        if self.objective.direction == "maximize":
            best = max(results, key=lambda x: x["value"])
        else:
            best = min(results, key=lambda x: x["value"])
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            best_params=best["params"],
            best_value=best["value"],
            all_trials=[{"trial_number": r["trial"], "params": r["params"], "value": r["value"]} 
                       for r in results],
            optimization_time_seconds=elapsed,
            n_trials=n_trials,
            objective_name=self.objective.name
        )
        
        self._results.append(result)
        self._save_result(result, "random")
        
        return result
    
    def _suggest_params(self, trial) -> Dict[str, Any]:
        """Generate parameters using Optuna trial."""
        params = {}
        for name, param in self.parameters.items():
            if param.param_type == "float":
                if param.log_scale:
                    params[name] = trial.suggest_float(name, param.low, param.high, log=True)
                elif param.step:
                    params[name] = trial.suggest_float(name, param.low, param.high, step=param.step)
                else:
                    params[name] = trial.suggest_float(name, param.low, param.high)
            elif param.param_type == "int":
                if param.step:
                    params[name] = trial.suggest_int(name, int(param.low), int(param.high), 
                                                    step=int(param.step))
                else:
                    params[name] = trial.suggest_int(name, int(param.low), int(param.high))
            elif param.param_type == "categorical":
                params[name] = trial.suggest_categorical(name, param.choices)
        
        return params
    
    def _random_params(self) -> Dict[str, Any]:
        """Generate random parameters."""
        params = {}
        for name, param in self.parameters.items():
            if param.param_type == "float":
                if param.log_scale:
                    log_val = np.random.uniform(np.log(param.low), np.log(param.high))
                    params[name] = np.exp(log_val)
                else:
                    params[name] = np.random.uniform(param.low, param.high)
            elif param.param_type == "int":
                params[name] = np.random.randint(int(param.low), int(param.high) + 1)
            elif param.param_type == "categorical":
                params[name] = np.random.choice(param.choices)
        
        return params
    
    def _generate_grid(self, points_per_param: int) -> List[Dict[str, Any]]:
        """Generate full grid of parameter combinations."""
        import itertools
        
        param_values = {}
        for name, param in self.parameters.items():
            if param.param_type == "float":
                if param.log_scale:
                    vals = np.logspace(np.log10(param.low), np.log10(param.high), points_per_param)
                else:
                    vals = np.linspace(param.low, param.high, points_per_param)
                param_values[name] = vals.tolist()
            elif param.param_type == "int":
                vals = np.linspace(param.low, param.high, points_per_param).astype(int)
                param_values[name] = list(set(vals))
            elif param.param_type == "categorical":
                param_values[name] = param.choices
        
        # Generate all combinations
        keys = list(param_values.keys())
        combinations = list(itertools.product(*[param_values[k] for k in keys]))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def sensitivity_analysis(
        self,
        base_params: Dict[str, Any],
        n_points: int = 10,
        variation_pct: float = 20.0
    ) -> Dict[str, SensitivityResult]:
        """
        Analyze how sensitive the objective is to each parameter.
        
        Args:
            base_params: Baseline parameter values
            n_points: Number of points to sample for each parameter
            variation_pct: Percentage variation around baseline
        
        Returns:
            Dict mapping parameter name to sensitivity results
        """
        results = {}
        base_objective = self.objective.evaluate(base_params)
        
        for name, param in self.parameters.items():
            if param.param_type == "categorical":
                # For categorical, test all choices
                values = param.choices
                test_values = [v for v in values if v != base_params.get(name)]
            else:
                # For numeric, vary around baseline
                base_val = base_params.get(name, (param.low + param.high) / 2)
                delta = base_val * (variation_pct / 100)
                
                low = max(param.low, base_val - delta)
                high = min(param.high, base_val + delta)
                
                if param.param_type == "int":
                    values = np.linspace(low, high, n_points).astype(int).tolist()
                else:
                    values = np.linspace(low, high, n_points).tolist()
                
                test_values = [v for v in values if v != base_params.get(name)]
            
            objective_values = []
            for val in test_values:
                test_params = base_params.copy()
                test_params[name] = val
                obj_val = self.objective.evaluate(test_params)
                objective_values.append(obj_val)
            
            # Calculate sensitivity score
            if test_values and objective_values:
                obj_range = max(objective_values) - min(objective_values)
                val_range = max(test_values) - min(test_values) if param.param_type != "categorical" else len(test_values)
                
                sensitivity = obj_range / val_range if val_range > 0 else 0
                
                # Parameter is stable if objective doesn't vary much
                is_stable = obj_range < abs(base_objective) * 0.1  # <10% variation
            else:
                sensitivity = 0
                is_stable = True
            
            results[name] = SensitivityResult(
                parameter_name=name,
                values=list(test_values),
                objective_values=objective_values,
                baseline_value=base_params.get(name),
                baseline_objective=base_objective,
                sensitivity_score=sensitivity,
                is_stable=is_stable
            )
        
        return results
    
    def parameter_importance(self) -> Dict[str, float]:
        """
        Calculate parameter importance based on optimization history.
        Requires Optuna and a completed optimization.
        """
        if not OPTUNA_AVAILABLE or self._study is None:
            logger.warning("Parameter importance requires Optuna and completed optimization")
            return {}
        
        try:
            importance = optuna.importance.get_param_importances(self._study)
            return dict(importance)
        except Exception as e:
            logger.error(f"Could not calculate importance: {e}")
            return {}
    
    def _save_result(self, result: OptimizationResult, method: str):
        """Save optimization result to file."""
        filename = f"opt_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved optimization results to {filepath}")


class WalkForwardOptimizer:
    """
    Walk-forward optimization with in-sample training and out-of-sample testing.
    Prevents overfitting by testing on unseen data.
    """
    
    def __init__(
        self,
        parameters: List[Parameter],
        train_objective: Callable,  # (params, train_data) -> metric
        test_objective: Callable,   # (params, test_data) -> metric
        data_splitter: Callable     # (data, fold) -> (train_data, test_data)
    ):
        self.parameters = parameters
        self.train_objective = train_objective
        self.test_objective = test_objective
        self.data_splitter = data_splitter
    
    def optimize(
        self,
        data: Any,
        n_folds: int = 5,
        trials_per_fold: int = 50,
        optimization_method: str = "bayesian"
    ) -> Dict:
        """
        Run walk-forward optimization.
        
        For each fold:
        1. Optimize on training set
        2. Test best params on test set
        
        Returns aggregate results showing true out-of-sample performance.
        """
        fold_results = []
        all_test_metrics = []
        
        for fold in range(n_folds):
            logger.info(f"Walk-forward fold {fold + 1}/{n_folds}")
            
            # Split data
            train_data, test_data = self.data_splitter(data, fold)
            
            # Create objective for training
            train_obj = BacktestObjective(
                backtest_runner=lambda p: self.train_objective(p, train_data),
                metric="sharpe_ratio",
                direction="maximize"
            )
            
            # Optimize on training data
            optimizer = ParameterOptimizer(self.parameters, train_obj)
            
            if optimization_method == "bayesian" and OPTUNA_AVAILABLE:
                result = optimizer.optimize_bayesian(n_trials=trials_per_fold, show_progress=False)
            else:
                result = optimizer.optimize_random(n_trials=trials_per_fold)
            
            # Test on out-of-sample data
            test_metric = self.test_objective(result.best_params, test_data)
            
            fold_results.append({
                "fold": fold,
                "train_metric": result.best_value,
                "test_metric": test_metric,
                "best_params": result.best_params,
                "overfitting_ratio": result.best_value / test_metric if test_metric != 0 else 0
            })
            
            all_test_metrics.append(test_metric)
        
        # Aggregate results
        avg_train = np.mean([r["train_metric"] for r in fold_results])
        avg_test = np.mean(all_test_metrics)
        std_test = np.std(all_test_metrics)
        
        return {
            "n_folds": n_folds,
            "avg_train_metric": avg_train,
            "avg_test_metric": avg_test,
            "std_test_metric": std_test,
            "overfitting_ratio": avg_train / avg_test if avg_test != 0 else 0,
            "fold_results": fold_results,
            "is_robust": avg_test > 0 and std_test < abs(avg_test) * 0.5  # Test metric is positive and stable
        }


# Common parameter sets for trading strategy optimization
STRATEGY_PARAMS = [
    Parameter("risk_per_trade", "float", 0.005, 0.03, step=0.005, default=0.01),
    Parameter("atr_multiplier", "float", 1.0, 3.0, step=0.25, default=1.5),
    Parameter("ema_fast", "int", 5, 15, default=9),
    Parameter("ema_slow", "int", 15, 30, default=21),
    Parameter("rsi_period", "int", 7, 21, default=14),
    Parameter("rsi_oversold", "int", 20, 40, default=30),
    Parameter("rsi_overbought", "int", 60, 80, default=70),
    Parameter("volume_multiplier", "float", 1.0, 3.0, step=0.25, default=1.5),
    Parameter("profit_target_rr", "float", 1.0, 3.0, step=0.25, default=1.5),
    Parameter("max_trades_per_day", "int", 1, 5, default=2),
]

ORB_PARAMS = [
    Parameter("orb_period_minutes", "categorical", choices=[15, 30, 45, 60], default=30),
    Parameter("orb_buffer_pct", "float", 0.001, 0.005, step=0.001, default=0.002),
    Parameter("orb_volume_filter", "float", 1.0, 2.5, step=0.25, default=1.5),
    Parameter("orb_range_filter_pct", "float", 0.3, 1.0, step=0.1, default=0.5),
]

VWAP_PARAMS = [
    Parameter("vwap_entry_sigma", "float", 1.5, 3.0, step=0.25, default=2.0),
    Parameter("vwap_exit_sigma", "float", 0.0, 1.0, step=0.25, default=0.5),
    Parameter("vwap_require_rsi", "categorical", choices=[True, False], default=True),
    Parameter("vwap_macd_confirm", "categorical", choices=[True, False], default=True),
]
