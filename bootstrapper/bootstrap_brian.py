#%%
from typing import Any, Callable, Dict, List, Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification import multiclass_auroc, multiclass_average_precision
from torchmetrics.wrappers.bootstrapping import _bootstrap_sampler
from monai.metrics import compute_dice
from monai.utils import ensure_tuple
from manafaln.core.builders import MetricV2Builder as MetricBuilder
from abc import ABC, abstractmethod
from torchmetrics.utilities import apply_to_collection


class BootStrapperBase(Metric, ABC):
    def __init__(
        self, 
        states: List[str], 
        metric: Metric, 
        num_bootstraps: int = 1000,
        quantile=[0.025, 0.975],
        num_classes = 3
        ):

        super().__init__()
        self.states = states
        for state in states:
            self.add_state(state, default=[], dist_reduce_fx='cat')
        self.metric = metric
        self.num_bootstraps = num_bootstraps
        self.quantile = quantile
        self.num_classes = num_classes
        self.metric_inputs = []

    def update(self, *args):
        for state, arg in zip(self.states, args):
            state_obj = getattr(self, state)
            state_obj.extend(arg)

    def compute(self) -> Dict[str, Tensor]:
        states = [getattr(self, state) for state in self.states]
        size = len(states[0])
        stack_temp = [torch.stack(getattr(self, state),dim=0) for state in self.states]

        output_dict = {}
        output_dict["all"] =  self.metric(*stack_temp)

        computed_vals = []
        for _ in range(self.num_bootstraps):
            bootstrapped = []
            idx = _bootstrap_sampler(size, sampling_strategy="multinomial").to(self.device)
            skip = False
            for element in stack_temp:
                bootstrap_temp = element.index_select(index=idx, dim=0)
                bootstrapped.append(bootstrap_temp) 
                if bootstrap_temp.dim()==1 and len(bootstrap_temp.unique())!=self.num_classes:
                    skip = True
                    break
            if skip:
                continue
            tmp = self.metric(*bootstrapped)
            computed_vals.append(tmp)

        computed_vals = torch.tensor(computed_vals, device=self.device)
        quantile = torch.tensor(self.quantile, device=self.device)
        output_quantiles = torch.quantile(computed_vals, quantile)
        for quantile, output_quantile in zip(self.quantile, output_quantiles):
            output_dict[f"quantile_{quantile}"] = output_quantile

        return output_dict


def AUC(pred_clf_ett, clf_ett, pred_clf_ngt, clf_ngt):
    num_classes=3
    auc1 = multiclass_auroc(pred_clf_ett, clf_ett, num_classes=num_classes)
    auc2 = multiclass_auroc(pred_clf_ngt, clf_ngt, num_classes=num_classes)
    out = (auc1+auc2)/2
    return out

def AP(pred_clf_ett, clf_ett, pred_clf_ngt, clf_ngt):
    num_classes=3
    auc1 = multiclass_average_precision(pred_clf_ett, clf_ett, num_classes=num_classes)
    auc2 = multiclass_average_precision(pred_clf_ngt, clf_ngt, num_classes=num_classes)
    out = (auc1+auc2)/2
    return out


pred_clf_ett = torch.tensor([[0.75, 0.15, 0.10],
                             [0.10, 0.75, 0.15],
                             [0.10, 0.15, 0.75],
                             [0.20, 0.75, 0.05],
                             [0.10, 0.05, 0.85],
                             [0.80, 0.13, 0.07],
                             [0.10, 0.83, 0.07]])
clf_ett = torch.tensor([0, 1, 2, 1, 2, 0, 1])
pred_clf_ngt = torch.tensor([[0.75, 0.15, 0.10],
                             [0.10, 0.75, 0.15],
                             [0.10, 0.15, 0.75],
                             [0.20, 0.75, 0.05],
                             [0.15, 0.05, 0.80],
                             [0.81, 0.15, 0.04],
                             [0.20, 0.73, 0.07]])
clf_ngt = torch.tensor([0, 1, 2, 0, 2, 0, 0])


seg_area = ['ETT', 'NGT', 'Trachea', 'Lung', 'Diaphragm', 'ETT_tip', 'NGT_tip', 'Carina']
clf_task = ['pred_clf_ett','clf_ett','pred_clf_ngt','clf_ngt']

my_metric = BootStrapperBase(states=['pred_clf_ett','clf_ett','pred_clf_ngt','clf_ngt'], metric=AUC)
my_metric.update(pred_clf_ett, clf_ett, pred_clf_ngt, clf_ngt)
my_metric.compute()
#%%
my_metric = BootStrapperBase(states=['pred_clf_ett','clf_ett','pred_clf_ngt','clf_ngt'], metric=AP)
my_metric.update(pred_clf_ett, clf_ett, pred_clf_ngt, clf_ngt)
my_metric.compute()
#%%

class CumulativeBootstrapper(Metric):
    """
    Compute the bootstrapped metrics for cumulative metric (e.g. AUC).

    Args:
        metric (Dict): The config to build the Metric object.
        num_bootstraps (int): Number of bootstrapping steps. Defaults to 1000.
        quantile (List[float]): The quantile of bootstrapped metrics to return. Defaults to [0.025, 0.975].
        sampling_strategy (Literal["poisson", "multinomial"]): Sampling strategy. Defaults to "multinomial".
    """
    full_state_update: bool = True
    def __init__(
        self,
        metric: Dict,
        num_bootstraps: int = 1000,
        quantile: List[float] = [0.025, 0.975],
        sampling_strategy: Literal["poisson", "multinomial"] = "multinomial",
        ):
        super().__init__()

        builder = MetricBuilder(check_instance=False)
        self.metric: Callable = builder(metric)
        self.num_bootstraps = num_bootstraps
        self.quantile = quantile
        self.sampling_strategy = sampling_strategy

    def update(self, *args: Any) -> None:
        """
        Update the states for the metric to be computed.
        """
        # If no state is present, initialize with the number of given args
        if self._defaults == {}:
            for idx in range(len(args)):
                self.add_state(f"state_{idx}", default=[], dist_reduce_fx='cat')

        # Update the states
        for state, arg in zip(self._defaults, args):
            getattr(self, state).extend(arg)

    def compute(self) -> Dict[str, Tensor]:
        """
        Computes the bootstrapped metric values.

        Returns:
            Dict[Tensor]: Results of bootstrapped metrics, containing the following keys:
                None: The metric as if no bootstrapping is done
                quantile_{q}: The quantile q of bootstrapped metrics

        Raise:
            ValueError: States have different number of length
        """

        states: List[list] = [getattr(self, state) for state in self._defaults]

        if len(set(len(state) for state in states)) != 1:
            raise ValueError("All states must have the same number of non-zero length")
        size = len(states[0])

        states: List[Tensor] = [torch.stack(state, dim=0) for state in states]

        output_dict = {}
        output_dict[None] = self.metric(*states)

        bootstrapped_metrics = []
        for _ in range(self.num_bootstraps):
            idx = _bootstrap_sampler(size, sampling_strategy=self.sampling_strategy).to(self.device)
            bootstrapped_states = [state.index_select(index=idx, dim=0) for state in states]
            bootstrapped_metric: Tensor = self.metric(*bootstrapped_states)
            if not torch.isnan(bootstrapped_metric):
                bootstrapped_metrics.append(bootstrapped_metric)

        if bootstrapped_metrics == []:
            for quantile in self.quantile:
                output_dict[f"quantile_{quantile}"] = torch.nan
            return output_dict

        bootstrapped_metrics = torch.tensor(bootstrapped_metrics, device=self.device)
        quantile = torch.tensor(self.quantile, device=self.device)
        output_quantiles = torch.quantile(bootstrapped_metrics, quantile)
        for quantile, output_quantile in zip(self.quantile, output_quantiles):
            output_dict[f"quantile_{quantile}"] = output_quantile

        return output_dict



#%%



