import flwr as fl
from model import get_evaluate_fn, get_model

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}

strategy = fl.server.strategy.FedProx(
    evaluate_fn=get_evaluate_fn(),
    on_fit_config_fn=lambda r: {"server_round": r},
    on_evaluate_config_fn=lambda r: {"server_round": r},
    fit_metrics_aggregation_fn=weighted_average,
    evaluate_metrics_aggregation_fn=weighted_average,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
    proximal_mu=0.1
)


fl.server.start_server(
    server_address="[::]:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
