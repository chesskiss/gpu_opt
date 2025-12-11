"""Run MPS vs CPU inference timing using modular components."""

from gpu_alloc import pick_device
from model import build_model
from eval import time_inference
import torch

from router import optimize

def run_once(model, device_preference: str, batch_size: int = 512, runs: int = 5) -> None:
    device = pick_device(device_preference)
    print(f"\nRunning on: {device}")

    avg_time = time_inference(model, device, batch_size=batch_size, runs=runs)
    print(f"Average forward pass time over {runs} runs: {avg_time:.4f} seconds")




def optimized_run(model, batch_size=512, runs=5):

    input_dim = getattr(model, "input_dim", 2048)

    # No device sent into optimize — it will route ops on its own
    model_fx = optimize(model, batch_size=batch_size, input_dim=input_dim)

    # Pick same default eval device (MPS or fallback)
    # device = pick_device("mps")
    device = torch.device(list(model_fx.device_map.values())[0])

    avg_time_fx = time_inference(model_fx, device, batch_size=batch_size, runs=runs)
    print(f"\nFX-Routed avg time: {avg_time_fx:.4f} seconds")



def main() -> None:
    print('MediumMLP speed Test')
    model = build_model('MediumMLP')
    run_once(model, "mps")
    run_once(model, "cpu")
    optimized_run(model)


    print('\n\n ToyHybrid speed Test')
    model = build_model('ToyHybrid')

    run_once(model, "mps")
    run_once(model, "cpu")
    optimized_run(model)



if __name__ == "__main__":
    main()
