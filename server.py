import logging
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy  # Corrected import
from flwr.common import FitIns, EvaluateRes  # Corrected import
from typing import List, Tuple, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(handler)

class CustomFedAvg(FedAvg):
    def configure_fit(self, server_round: int, parameters, client_manager) -> List[Tuple[ClientProxy, FitIns]]:
        """Adjust min clients based on round."""
        if server_round <= 3:
            self.min_available_clients = 2
            self.min_fit_clients = 2
        else:
            self.min_available_clients = 3
            self.min_fit_clients = 3
        logger.debug(f"Round {server_round}: Configured min_fit_clients={self.min_fit_clients}")
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],  # Fixed type hint
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Log accuracy and stop if not enough clients."""
        if server_round <= 3 and len(results) >= 2:
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)
            accuracy = np.mean([r.metrics["accuracy"] for _, r in results])
            logger.info(f"Round {server_round}: Aggregated accuracy = {accuracy:.4f}")
            return loss, {"accuracy": accuracy}
        elif server_round > 3 and len(results) < 3:
            logger.error(f"Round {server_round}: Only {len(results)} clients available, need 3. Stopping training.")
            accuracy = np.mean([r.metrics["accuracy"] for _, r in results]) if results else 0.0
            logger.info(f"Final aggregated accuracy before stopping: {accuracy:.4f}")
            raise SystemExit("Not enough clients to continue training.")
        return super().aggregate_evaluate(server_round, results, failures)

def main():
    logger.info("Starting Flower server")
    strategy = CustomFedAvg(
        min_available_clients=2,  # Start with 2 clients
        min_fit_clients=2,
        min_evaluate_clients=2,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),  # 4 rounds total
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
