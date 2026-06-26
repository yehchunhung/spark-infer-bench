"""
Adapted from https://github.com/ray-project/ray/blob/6fe3e7657436317ed6214badd26ca860f5346cea/doc/source/llm/doc_code/serve/multi_gpu/dp_pd_example.py

This file serves as a documentation example and CI test for data parallel + prefill-decode disaggregation.

Structure:
1. Monkeypatch setup: Ensures serve.run is non-blocking and removes accelerator requirements for CI testing.
2. Docs example (between __dp_pd_example_start/end__): Embedded in Sphinx docs via literalinclude.
3. Test validation (deployment status polling + cleanup)
"""

import argparse
import time
from ray import serve
from ray.serve.schema import ApplicationStatus
from ray.serve._private.constants import SERVE_DEFAULT_APP_NAME
from ray.serve import llm
from ray.serve.llm.deployment import PDProxyServer
from ray.serve.llm.ingress import OpenAiIngress, make_fastapi_ingress
from ray.serve.llm import LLMConfig, build_llm_deployment # > 2.51.1: build_dp_deployment

# Check if NIXL is available (required for NixlConnector)
try:
    import nixl  # noqa: F401
    NIXL_AVAILABLE = True
except ImportError:
    NIXL_AVAILABLE = False

if not NIXL_AVAILABLE:
    raise ImportError(
        "NIXL is required for this example but is not installed. "
        "Install it with: pip install nixl or uv pip install nixl"
    )

_original_serve_run = serve.run
_original_build_dp_deployment = llm.build_llm_deployment # llm.build_dp_deployment


def _non_blocking_serve_run(app, **kwargs):
    """Forces blocking=False for testing"""
    kwargs["blocking"] = False
    return _original_serve_run(app, **kwargs)


def _testing_build_dp_deployment(llm_config, **kwargs):
    """Removes accelerator requirements for testing"""
    if llm_config.accelerator_type is not None:
        llm_config.accelerator_type = None
    return _original_build_dp_deployment(llm_config, **kwargs)


serve.run = _non_blocking_serve_run
llm.build_dp_deployment = _testing_build_dp_deployment


def create_llm_config(model_name, data_parallel_size, tensor_parallel_size, max_model_len, max_num_seqs, dp_size_per_node=2):
    """Create LLMConfig with specified parameters"""
    return LLMConfig(
        model_loading_config={
            "model_id": model_name
        },
        engine_kwargs={
            "data_parallel_size": data_parallel_size,
            "tensor_parallel_size": tensor_parallel_size,
            "kv_transfer_config": {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_both",
            },
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
        },
        experimental_configs={
            "dp_size_per_node": dp_size_per_node,
        },
    )


def deploy_application(prefill_config, decode_config, blocking=True):
    """Deploy the Ray Serve application with the given configurations"""
    # Build prefill and decode deployments with DP
    prefill_deployment = build_llm_deployment(prefill_config, name_prefix="Prefill:")
    decode_deployment = build_llm_deployment(decode_config, name_prefix="Decode:")

    # Create PDProxyServer to coordinate between prefill and decode
    proxy_options = PDProxyServer.get_deployment_options(prefill_config, decode_config)
    proxy_deployment = serve.deployment(PDProxyServer).options(**proxy_options).bind(
        prefill_server=prefill_deployment,
        decode_server=decode_deployment,
    )

    # Create OpenAI-compatible ingress
    ingress_options = OpenAiIngress.get_deployment_options([prefill_config, decode_config])
    ingress_cls = make_fastapi_ingress(OpenAiIngress)
    ingress_deployment = serve.deployment(ingress_cls).options(**ingress_options).bind(
        llm_deployments=[proxy_deployment]
    )

    # Deploy the application
    serve.run(ingress_deployment, blocking=blocking)


def wait_for_deployment(timeout_seconds=300):
    """Wait for the deployment to reach RUNNING status"""
    status = ApplicationStatus.NOT_STARTED
    start_time = time.time()

    while (
        status != ApplicationStatus.RUNNING and time.time() - start_time < timeout_seconds
    ):
        status = serve.status().applications[SERVE_DEFAULT_APP_NAME].status

        if status in [ApplicationStatus.DEPLOY_FAILED, ApplicationStatus.UNHEALTHY]:
            raise AssertionError(f"Deployment failed with status: {status}")

        time.sleep(1)

    if status != ApplicationStatus.RUNNING:
        raise AssertionError(
            f"Deployment failed to reach RUNNING status within {timeout_seconds}s. Current status: {status}"
        )


def main():
    parser = argparse.ArgumentParser(description='Ray Serve PD Disaggregation Deployment')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Model ID to use (default: Qwen/Qwen2.5-0.5B-Instruct)')
    parser.add_argument('--prefill-dp-size', type=int, default=1,
                        help='Data parallel size for prefill (default: 1)')
    parser.add_argument('--decode-dp-size', type=int, default=1,
                        help='Data parallel size for decode (default: 1)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Tensor parallel size (default: 1)')
    parser.add_argument('--max-model-len', type=int, default=1024,
                        help='Maximum model length (default: 1024)')
    parser.add_argument('--max-num-seqs', type=int, default=32,
                        help='Maximum number of sequences (default: 32)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Deployment timeout in seconds (default: 300)')
    parser.add_argument('--blocking', action='store_true', default=True,
                        help='Run serve.run in blocking mode (default: True)')
    parser.add_argument('--no-blocking', dest='blocking', action='store_false',
                        help='Run serve.run in non-blocking mode')
    
    args = parser.parse_args()

    # Create configurations
    prefill_config = create_llm_config(
        model_name=args.model_name,
        data_parallel_size=args.prefill_dp_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs
    )
    
    decode_config = create_llm_config(
        model_name=args.model_name,
        data_parallel_size=args.decode_dp_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs
    )

    # Deploy application
    deploy_application(prefill_config, decode_config, blocking=args.blocking)
    
    # Wait for deployment if in blocking mode
    if args.blocking:
        wait_for_deployment(args.timeout)
        print("Deployment successful!")
        serve.shutdown()


if __name__ == "__main__":
    main()