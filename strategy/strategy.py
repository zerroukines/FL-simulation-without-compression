import logging
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import EvaluateRes, FitRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from prometheus_client import Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FedCustom(fl.server.strategy.FedAvg):
    def __init__(
        self,
        accuracy_gauge: Gauge = None,
        loss_gauge: Gauge = None,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        *args,
        **kwargs
    ):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            *args,
            **kwargs
        )
        self.accuracy_gauge = accuracy_gauge
        self.loss_gauge = loss_gauge

    def __repr__(self) -> str:
        return "FedCustom"

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training, tolerating some client unavailability."""
        config = {"server_round": server_round}
        sample_size = max(self.min_fit_clients, int(client_manager.num_available() * 0.5))  # At least 50%
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_available_clients
        )
        logger.info(f"Round {server_round}: Configured {len(clients)} clients for training")
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:
        """Aggregate training results, tolerating some failures."""
        if not results:
            logger.warning(f"Round {server_round}: No fit results to aggregate")
            return None, {}

        # Use FedAvg's aggregate function for parameter averaging
        aggregated_params = aggregate(
            [(fit_res.parameters, fit_res.num_examples) for _, fit_res in results]
        )

        # Aggregate accuracy from client metrics
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        total_examples = sum([r.num_examples for _, r in results])
        accuracy_aggregated = sum(accuracies) / total_examples if total_examples > 0 else 0

        self.accuracy_gauge.set(accuracy_aggregated)
        logger.info(f"Round {server_round}: Aggregated fit accuracy = {accuracy_aggregated:.4f}")

        return aggregated_params, {"accuracy": accuracy_aggregated}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """Configure the next round of evaluation."""
        config = {"server_round": server_round}
        sample_size = max(self.min_evaluate_clients, int(client_manager.num_available() * 0.5))
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_available_clients
        )
        logger.info(f"Round {server_round}: Configured {len(clients)} clients for evaluation")
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses and accuracy using weighted average."""
        if not results:
            logger.warning(f"Round {server_round}: No evaluation results to aggregate")
            return None, {}

        # Calculate weighted average for loss
        loss_aggregated = weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )

        # Calculate weighted average for accuracy
        accuracies = [
            evaluate_res.metrics["accuracy"] * evaluate_res.num_examples
            for _, evaluate_res in results
        ]
        examples = [evaluate_res.num_examples for _, evaluate_res in results]
        accuracy_aggregated = sum(accuracies) / sum(examples) if sum(examples) > 0 else 0

        # Update Prometheus gauges
        self.accuracy_gauge.set(accuracy_aggregated)
        self.loss_gauge.set(loss_aggregated)

        metrics_aggregated = {"loss": loss_aggregated, "accuracy": accuracy_aggregated}
        logger.info(
            f"Round {server_round}: Eval - Loss = {loss_aggregated:.4f}, Accuracy = {accuracy_aggregated:.4f}"
        )

        return loss_aggregated, metrics_aggregated


# Example instantiation (for testing outside server.py)
if __name__ == "__main__":
    strategy = FedCustom(
        accuracy_gauge=Gauge("model_accuracy", "Test accuracy"),
        loss_gauge=Gauge("model_loss", "Test loss"),
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    logger.info(f"Strategy initialized: {strategy}")