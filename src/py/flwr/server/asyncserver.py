import concurrent.futures
import time
import timeit
from logging import DEBUG, INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Reconnect,
    Scalar,
    Weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy, FedAsync
import queue

DEPRECATION_WARNING_EVALUATE = """
DEPRECATION WARNING: Method

    Server.evaluate(self, rnd: int) -> Optional[
        Tuple[Optional[float], EvaluateResultsAndFailures]
    ]

is deprecated and will be removed in a future release, use

    Server.evaluate_round(self, rnd: int) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]

instead.
"""

DEPRECATION_WARNING_EVALUATE_ROUND = """
DEPRECATION WARNING: The configured Strategy uses a deprecated aggregate_evaluate
return format:

    Strategy.aggregate_evaluate(...) -> Optional[float]

This format is deprecated and will be removed in a future release. It should use

    Strategy.aggregate_evaluate(...) -> Tuple[Optional[float], Dict[str, Scalar]]

instead.
"""

DEPRECATION_WARNING_FIT_ROUND = """
DEPRECATION WARNING: The configured Strategy uses a deprecated aggregate_fit
return format:

    Strategy.aggregate_fit(...) -> Optional[Weights]

This format is deprecated and will be removed in a future release. It should use

    Strategy.aggregate_fit(...) -> Tuple[Optional[Weights], Dict[str, Scalar]]

instead.
"""

FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]], List[BaseException]
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, Disconnect]], List[BaseException]
]

queue = queue.Queue()
round_time = []
eval_time = []


class AsyncServer:
    """Flower async_server."""

    def __init__(
            self, client_manager: ClientManager, strategy: Optional[Strategy] = None, alpha=0.5, flag=True,
    ) -> None:
        # print('server init')
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAsync()
        self.current_round = 0
        self.alpha = alpha
        self.flag = flag

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        # print('server set_strategy')
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int) -> History:
        """Run federated averaging for a number of rounds."""
        # print('server fit')
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters()
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_metrics_centralized(rnd=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        self.async_fit_rounds(history, num_rounds, start_time)
        return history

        # for current_round in range(1, num_rounds + 1):
        #     # Train model and replace previous global model
        #     res_fit = self.fit_round(rnd=current_round)
        #     if res_fit:
        #         parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
        #         if parameters_prime:
        #             self.parameters = parameters_prime
        #
        #     # Evaluate model using strategy implementation
        #     res_cen = self.strategy.evaluate(parameters=self.parameters)
        #     if res_cen is not None:
        #         loss_cen, metrics_cen = res_cen
        #         log(
        #             INFO,
        #             "fit progress: (%s, %s, %s, %s)",
        #             current_round,
        #             loss_cen,
        #             metrics_cen,
        #             timeit.default_timer() - start_time,
        #         )
        #         history.add_loss_centralized(rnd=current_round, loss=loss_cen)
        #         history.add_metrics_centralized(rnd=current_round, metrics=metrics_cen)
        #
        #     # Evaluate model on a sample of available clients
        #     res_fed = self.evaluate_round(rnd=current_round)
        #     if res_fed:
        #         loss_fed, evaluate_metrics_fed, _ = res_fed
        #         if loss_fed:
        #             history.add_loss_distributed(rnd=current_round, loss=loss_fed)
        #             history.add_metrics_distributed(
        #                 rnd=current_round, metrics=evaluate_metrics_fed
        #             )
        #
        # # Bookkeeping
        # end_time = timeit.default_timer()
        # elapsed = end_time - start_time
        # log(INFO, "FL finished in %s", elapsed)
        # return history

    def async_fit_rounds(
            self,
            history: History,
            num_rounds: int,
            start_time
    ) -> History:
        """Perform asynchronous federated optimization."""
        # Run federated learning for num_rounds
        # print('server fit rounds')
        log(INFO, "FedAsync starting")

        # 刚开始让所有的clients都加入training
        client_instructions = self.strategy.configure_fit(
            rnd=self.current_round, parameters=self.parameters, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "fit_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "fit_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from the queue
        # 创建一个queue，当一个client完成返回后，加入队列
        # 每次server从队列当中拿出一个，收完他的parameter后，更新global的parameter
        # server直接让这个client基于新的parameter继续下去训练，不需要等其它人
        flag = True
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
        [executor.submit(fit_client, c, ins) for c, ins in client_instructions]
        while flag:
            # print(f'queue length {queue.qsize()}')
            if not queue.empty():
                results: List[Tuple[ClientProxy, FitRes]] = []
                result = queue.get()
                results.append(result)
                # print(' current round ', str(self.current_round))
                # print('num round', str(num_rounds))
                self.current_round += 1

                # Aggregate training results
                aggregated_result: Union[
                    Tuple[Optional[Parameters], Dict[str, Scalar]],
                    Optional[Weights],  # Deprecated
                ] = self.strategy.weighted_aggregate_fit(rnd=self.current_round, alpha=self.alpha,
                                                         gl_parameters=self.parameters,
                                                         results=results, failures=[])

                if aggregated_result is not None:
                    parameters_aggregated, metrics_aggregated = aggregated_result
                    self.parameters = parameters_aggregated
                # strategy 要新加的function，直接返回当前的(client, FitIns)的tuple
                client_instructions = self.strategy.configure_fit_one(
                    rnd=self.current_round, parameters=self.parameters, client=result[0]
                )
                s = time.time()
                future = executor.submit(fit_client, client_instructions[0], client_instructions[1])
                # future.exception()

                # print(time.time() - s)
                # Evaluate model using strategy implementation
                # if self.current_round % 10 == 0:
                if self.current_round % 5 == 0 or self.current_round == num_rounds:
                # if True:
                    res_cen = self.strategy.evaluate(parameters=self.parameters)
                    # self.current_round = int(self.current_round / 10)
                    if res_cen is not None:
                        loss_cen, metrics_cen = res_cen
                        t = timeit.default_timer() - start_time
                        log(
                            INFO,
                            "fit progress: (%s, %s, %s, %s)",
                            # int(self.current_round / 10),
                            int(self.current_round),
                            loss_cen,
                            metrics_cen,
                            t,
                        )
                        history.add_loss_centralized(rnd=int(self.current_round), loss=loss_cen)
                        history.add_metrics_centralized(rnd=int(self.current_round), metrics=metrics_cen)
                        history.add_timer_acc_centralized(rnd=int(self.current_round), t=t, acc=metrics_cen['accuracy'])
                        history.add_timer_losses_centralized(rnd=int(self.current_round), t=t, loss=loss_cen)

                    # local evaluation
                if not self.flag:
                    print(future.exception())
                    # res_fed = self.evaluate_round_one(rnd=self.current_round, client=result[0])
                    res_fed = self.evaluate_round(rnd=self.current_round)
                    if res_fed:
                        loss_fed, evaluate_metrics_fed, _ = res_fed
                        if loss_fed:
                            history.add_loss_distributed(rnd=int(self.current_round), loss=loss_fed)
                            history.add_metrics_distributed(
                                rnd=int(self.current_round), metrics=evaluate_metrics_fed
                            )
                else:
                    if self.current_round == 1 or self.current_round % 10 == 0 or self.current_round == num_rounds:
                        # if True:
                        print(future.exception())
                        # res_fed = self.evaluate_round_one(rnd=self.current_round, client=result[0])
                        res_fed = self.evaluate_round(rnd=self.current_round)
                        if res_fed:
                            loss_fed, evaluate_metrics_fed, _ = res_fed
                            if loss_fed:
                                history.add_loss_distributed(rnd=int(self.current_round / 10), loss=loss_fed)
                                history.add_metrics_distributed(
                                    rnd=int(self.current_round / 10), metrics=evaluate_metrics_fed
                                )
                # print(future.exception())

            # print('num rounds', str(num_rounds))
            if self.current_round == num_rounds:
                flag = False
                total = 0
                for item in round_time:
                    total += item
                print('total time used in fit transmitting ', total)
                total = 0
                for item in eval_time:
                    total += item
                print('total time used in eval transmitting ', total)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate(
            self, rnd: int
    ) -> Optional[Tuple[Optional[float], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""
        # print('server eval')

        log(WARNING, DEPRECATION_WARNING_EVALUATE)
        res = self.evaluate_round(rnd)
        if res is None:
            return None
        # Deconstruct
        loss, _, results_and_failures = res
        return loss, results_and_failures

    def evaluate_round_one(
            self, rnd: int, client: ClientProxy
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""
        # print('server eval_round')

        # Get clients and their respective instructions from strategy
        client, ins = self.strategy.configure_evaluate_one(
            rnd=rnd, parameters=self.parameters, client=client
        )
        if not client:
            log(INFO, "evaluate_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "evaluate_round: strategy sampled %s clients (out of %s)",
            1,
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        cl, eval_res = evaluate_client(client, ins)
        log(
            DEBUG,
            "evaluate_round",
        )
        results = []
        results.append((cl, eval_res))

        # Aggregate the evaluation results
        aggregated_result: Union[
            Tuple[Optional[float], Dict[str, Scalar]],
            Optional[float],  # Deprecated
        ] = self.strategy.aggregate_evaluate(rnd, results, [])

        metrics_aggregated: Dict[str, Scalar] = {}
        if aggregated_result is None:
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_EVALUATE_ROUND)
            loss_aggregated = None
        elif isinstance(aggregated_result, float):
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_EVALUATE_ROUND)
            loss_aggregated = aggregated_result
        else:
            loss_aggregated, metrics_aggregated = aggregated_result

        return loss_aggregated, metrics_aggregated, (results, [])

    def evaluate_round(
            self, rnd: int
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""
        # print('server eval_round')

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "evaluate_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "evaluate_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(client_instructions)
        log(
            DEBUG,
            "evaluate_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Union[
            Tuple[Optional[float], Dict[str, Scalar]],
            Optional[float],  # Deprecated
        ] = self.strategy.aggregate_evaluate(rnd, results, failures)
        # print('agg', aggregated_result)

        metrics_aggregated: Dict[str, Scalar] = {}
        if aggregated_result is None:
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_EVALUATE_ROUND)
            loss_aggregated = None
        elif isinstance(aggregated_result, float):
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_EVALUATE_ROUND)
            loss_aggregated = aggregated_result
        else:
            loss_aggregated, metrics_aggregated = aggregated_result

        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
            self, rnd: int
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # print('server fit')

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )

        if not client_instructions:
            log(INFO, "fit_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "fit_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(client_instructions)
        log(
            DEBUG,
            "fit_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Union[
            Tuple[Optional[Parameters], Dict[str, Scalar]],
            Optional[Weights],  # Deprecated
        ] = self.strategy.aggregate_fit(rnd, results, failures)

        metrics_aggregated: Dict[str, Scalar] = {}
        if aggregated_result is None:
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_FIT_ROUND)
            parameters_aggregated = None
        elif isinstance(aggregated_result, list):
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_FIT_ROUND)
            parameters_aggregated = weights_to_parameters(aggregated_result)
        else:
            parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, (results, failures)

    def disconnect_all_clients(self) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        _ = shutdown(clients=[all_clients[k] for k in all_clients.keys()])

    def _get_initial_parameters(self) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # print('server get_init_params')
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        parameters_res = random_client.get_parameters()
        log(INFO, "Received initial parameters from one random client")
        return parameters_res.parameters


def shutdown(clients: List[ClientProxy]) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    reconnect = Reconnect(seconds=None)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(reconnect_client, c, reconnect) for c in clients]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, Disconnect]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
        client: ClientProxy, reconnect: Reconnect
) -> Tuple[ClientProxy, Disconnect]:
    """Instruct a single client to disconnect and (optionally) reconnect
    later."""
    disconnect = client.reconnect(reconnect)
    return client, disconnect


def fit_clients(
        client_instructions: List[Tuple[ClientProxy, FitIns]]
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    # print('server fit client sssssssss')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fit_client, c, ins) for c, ins in client_instructions
        ]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def fit_client(client: ClientProxy, ins: FitIns) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    # print('server fit client')
    t = time.time()
    fit_res = client.fit(ins)
    t = time.time() - t
    round_time.append(t - fit_res.metrics['fit_time'])
    queue.put((client, fit_res))
    # print(f'queue size {queue.qsize()}')
    return client, fit_res


def evaluate_clients(
        client_instructions: List[Tuple[ClientProxy, EvaluateIns]]
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    # print('server eval client   sssssss')
    t = time.time()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate_client, c, ins) for c, ins in client_instructions
        ]
        concurrent.futures.wait(futures)
    t = time.time() - t
    e_time = []
    total = 0
    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            e_time.append(t - result[1].metrics["eval_time"])
            results.append(result)
    for t in e_time:
        total += t
    eval_time.append(total / len(results))
    return results, failures


def evaluate_client(
        client: ClientProxy, ins: EvaluateIns
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    # print('server eval client')
    evaluate_res = client.evaluate(ins)
    return client, evaluate_res

# Async里最主要的function，收到一个client的parameter直接更新并让这个client
# 拿着新的parameter继续训练，不需要等其它clients


# def async_fit_rounds(
#         self,
#         history: History,
#         num_rounds: int,
#         start_time
# ) -> History:
#     """Perform asynchronous federated optimization."""
#     # Run federated learning for num_rounds
#     log(INFO, "FedAsync starting")
#
#     # 刚开始让所有的clients都加入training
#     client_instructions = self.strategy.configure_fit(
#         rnd=self.current_round, parameters=self.parameters, client_manager=self._client_manager
#     )
#     if not client_instructions:
#         log(INFO, "fit_round: no clients selected, cancel")
#         return None
#     log(
#         DEBUG,
#         "fit_round: strategy sampled %s clients (out of %s)",
#         len(client_instructions),
#         self._client_manager.num_available(),
#     )
#
#     # Collect `fit` results from the queue
#     # 创建一个queue，当一个client完成返回后，加入队列
#     # 每次server从队列当中拿出一个，收完他的parameter后，更新global的parameter
#     # server直接让这个client基于新的parameter继续下去训练，不需要等其它人
#     flag = True
#     executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
#     [executor.submit(fit_client, c, ins) for c, ins in client_instructions]
#     while flag:
#         if not queue.empty():
#             results: List[Tuple[ClientProxy, FitRes]] = []
#             result = queue.get()
#             results.append(result)
#             self.current_round += 1
#
#             # Aggregate training results
#             aggregated_result: Union[
#                 Tuple[Optional[Parameters], Dict[str, Scalar]],
#                 Optional[Weights],  # Deprecated
#             ] = self.strategy.weighted_aggregate_fit(rnd=self.current_round,
#                                                      gl_parameters=self.parameters,
#                                                      results=results)
#
#             if aggregated_result is not None:
#                 parameters_aggregated, metrics_aggregated = aggregated_result
#                 self.parameters = parameters_aggregated
#             # strategy 要新加的function，直接返回当前的(client, FitIns)的tuple
#             client_instructions = self.strategy.configure_fit_one(
#                 rnd=self.current_round, parameters=self.parameters, client=result[0]
#             )
#             executor.submit(fit_client, client_instructions[0], client_instructions[1])
#
#             # Evaluate model using strategy implementation
#             if self.current_round % 10 == 0:
#                 res_cen = self.strategy.evaluate(parameters=self.parameters)
#                 if res_cen is not None:
#                     loss_cen, metrics_cen = res_cen
#                     log(
#                         INFO,
#                         "fit progress: (%s, %s, %s, %s)",
#                         int(self.current_round / 10),
#                         loss_cen,
#                         metrics_cen,
#                         timeit.default_timer() - start_time,
#                     )
#                     history.add_loss_centralized(rnd=self.current_round, loss=loss_cen)
#                     history.add_metrics_centralized(rnd=self.current_round, metrics=metrics_cen)
#
#                     # local evaluation
#                     # self.evaluate_round(rnd=self.current_round)
#
#         if self.current_round == num_rounds:
#             flag = False
#
#     # Bookkeeping
#     end_time = timeit.default_timer()
#     elapsed = end_time - start_time
#     log(INFO, "FL finished in %s", elapsed)
#     return history


# FedAsync
# strategy:


# def configure_fit_one(
#         self, rnd: int, parameters: Parameters, client: ClientProxy
# ) -> Tuple[ClientProxy, FitIns]:
#     """Configure the next round of training."""
#     config = {}
#     config = {}
#     global_round = rnd
#     if self.on_fit_config_fn is not None:
#         # Custom fit config function provided
#         config = self.on_fit_config_fn(rnd)
#     fit_ins = FitIns(parameters, config)
#     return client, fit_ins


# # Aggregation:
#
#
# def aggregate_async(
#         gl_weights: Weights, results: List[Tuple[Weights, int]], alpha: float
# ) -> Weights:
#     """Update global model by weighted aggregation"""
#
#     for weights, progress in results:
#         c_weights = weights
#
#     c_weights_a = [each * alpha for each in c_weights]
#     gl_weights_a = [each * (1 - alpha) for each in gl_weights]
#
#     """ w_g(t) = (1-alpha)* w_g(t-1) + alpha * w_k(t-1)"""
#     weights_prime: Weights = [
#         reduce(np.add, layer_updates)
#         for layer_updates in zip(gl_weights_a, c_weights_a)
#     ]
#     return weights_prime
