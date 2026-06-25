from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def generate_launch_description():
    """Legacy alias for the SPU-BERT single-human test launch file."""
    launch_path = Path(__file__).with_name("spubert_single_human_test.launch.py")
    spec = spec_from_file_location("spubert_single_human_test_launch", launch_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load launch file: {launch_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()
