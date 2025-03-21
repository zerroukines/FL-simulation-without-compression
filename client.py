import argparse
import logging
import os
import sys
import signal
import psutil
import torch
import flwr as fl
import numpy as np
from torch.utils.data import DataLoader
from helpers.load_data import load_data
from model.model import Model  # Your custom Model class

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logger.handlers = [FlushHandler(sys.stdout)]
logger.handlers[0].setFormatter(formatter)

def signal_handler(sig, frame):
    logger.error(f"Client {args.client_id}: Killed by signal {signal.Signals(sig).name}, likely due to memory limit")
    sys.exit(137)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser(description="Flower client")
parser.add_argument("--server_address", type=str, default="server:8080")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--client_id", type=int, default=1)
parser.add_argument("--total_clients", type=int, default=5)
parser.add_argument("--data_percentage", type=float, default=0.5)
args = parser.parse_args()

logger.debug(f"Client {args.client_id}: Starting client process, PID={os.getpid()}")
logger.debug(f"Client {args.client_id}: Memory usage at startup: {psutil.virtual_memory().used / 1024**2:.2f} MB")

logger.debug(f"Client {args.client_id}: Initializing model with LR={args.learning_rate}")
model = Model(learning_rate=args.learning_rate)  # Your vit_h_14 model
logger.debug(f"Client {args.client_id}: Model initialized, Memory usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")

device = torch.device("cpu")
model.get_model().to(device)
logger.debug(f"Client {args.client_id}: Model moved to {device}, Memory usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")

class Client(fl.client.NumPyClient):
    def __init__(self, args):
        self.args = args
        logger.debug(f"Client {self.args.client_id}: Starting data preparation")
        self.train_loader, self.test_loader = load_data(
            data_sampling_percentage=self.args.data_percentage,
            client_id=self.args.client_id,
            total_clients=self.args.total_clients,
        )
        logger.debug(f"Client {self.args.client_id}: Data prepared - Train size: {len(self.train_loader.dataset)}, Memory usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")

    def get_parameters(self, config):
        logger.debug(f"Client {self.args.client_id}: Getting model parameters")
        return [param.detach().cpu().numpy() for param in model.get_model().parameters()]

    def fit(self, parameters, config):
        logger.debug(f"Client {self.args.client_id}: Starting fit, Round {config.get('round', 'unknown')}")
        self._set_parameters(parameters)
        model.get_model().train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            logger.debug(f"Client {self.args.client_id}: Processing batch {batch_idx+1}, Memory usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")
            data, target = data.to(device), target.to(device)
            model.optimizer.zero_grad()
            output = model.forward(data)
            loss = model.loss_function(output, target)
            loss.backward()
            model.optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        accuracy = correct / total
        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Client {self.args.client_id}: Fit completed - Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        return self.get_parameters(config), len(self.train_loader.dataset), {"accuracy": accuracy}

    def evaluate(self, parameters, config):
        logger.debug(f"Client {self.args.client_id}: Starting evaluate")
        self._set_parameters(parameters)
        model.get_model().eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = model.forward(data)
                total_loss += model.loss_function(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total
        logger.info(f"Client {self.args.client_id}: Eval - Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        return avg_loss, len(self.test_loader.dataset), {"accuracy": accuracy}

    def _set_parameters(self, parameters):
        logger.debug(f"Client {self.args.client_id}: Setting parameters")
        for param, new_param in zip(model.get_model().parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=torch.float32).to(device)

def start_fl_client():
    try:
        logger.debug(f"Client {args.client_id}: Creating Flower client")
        client = Client(args).to_client()
        logger.debug(f"Client {args.client_id}: Connecting to server at {args.server_address}")
        fl.client.start_client(server_address=args.server_address, client=client)
    except Exception as e:
        logger.error(f"Client {args.client_id}: Error starting FL client: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    logger.debug(f"Client {args.client_id}: Main execution started")
    start_fl_client()