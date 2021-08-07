import os
import numpy as np
import pickle
from dataclasses import dataclass
import PIL

from noise_reducers.binary_gibbs_noise_reducer import BinaryGibbsNoiseReducer
from noise_reducers.binary_gradient_noise_reducer import BinaryGradientNoiseReducer
from noise_reducers import image_utils


@dataclass
class Experiment(object):
    experiment_name: str
    noise_level: float


EXPERIMENTS = [
    # Experiment(experiment_name="100x100_medium_noise10", noise_level=0.1),
    # Experiment(experiment_name="300x300_medium_noise10", noise_level=0.1),
    # Experiment(experiment_name="1000x1000_medium_noise10", noise_level=0.1),
    # Experiment(experiment_name="300x300_easy_noise10", noise_level=0.1),
    # Experiment(experiment_name="300x300_medium_noise10", noise_level=0.1),
    # Experiment(experiment_name="300x300_hard_noise10", noise_level=0.1),
    # Experiment(experiment_name="300x300_medium_noise5", noise_level=0.05),
    # Experiment(experiment_name="300x300_medium_noise10", noise_level=0.1),
    # Experiment(experiment_name="300x300_medium_noise15", noise_level=0.15),
    Experiment(experiment_name="300x300_medium_noise15", noise_level=0.01),
    Experiment(experiment_name="300x300_medium_noise15", noise_level=0.05),
    Experiment(experiment_name="300x300_medium_noise15", noise_level=0.15),
    Experiment(experiment_name="300x300_medium_noise15", noise_level=0.25),
    Experiment(experiment_name="300x300_medium_noise15", noise_level=0.35),
]
IMAGES_PATH = "../binary_images"
IMAGES_PER_EXPERIMENT = 30
ITERATIONS_PER_EXPERIMENT = 50_000
ITERATIONS_PER_EVALUATION = 1000


@dataclass
class ImagePair(object):
    image_id: int
    ground_truth: np.ndarray
    observation: np.ndarray


def get_image_pairs_to_evaluate(experiment_name):
    images_path = os.path.join(IMAGES_PATH, experiment_name)
    image_pairs = []
    for image_id in range(IMAGES_PER_EXPERIMENT):
        ground_truth_path = os.path.join(images_path, f"image_{image_id}_ground_truth.png")
        observation_path = os.path.join(images_path, f"image_{image_id}_observation.png")
        image_pairs.append(ImagePair(
            image_id=image_id,
            ground_truth=image_utils.load_binary_image_as_numpy_array(ground_truth_path),
            observation=image_utils.load_binary_image_as_numpy_array(observation_path),
        ))
    return image_pairs


def run_with_reducer(reducer, experiment_name, storage_folder):
    os.makedirs(storage_folder, exist_ok=True)
    for image_pair in get_image_pairs_to_evaluate(experiment_name):
        reduction_result = reducer.reduce_noise(
            original_image=image_pair.ground_truth, observation=image_pair.observation,
        )
        reduced_image = PIL.Image.fromarray(reduction_result.reduced_image.astype(np.uint8))
        reduced_image.save(os.path.join(storage_folder, f"reduced_image_{image_pair.image_id}.png"), format="PNG")
    with open(os.path.join(storage_folder, "average_statistics.pickle"), mode="wb") as file_stream:
        pickle.dump(reducer.average_statistics, file_stream)


def run_with_gibbs_reducer(experiment):
    reducer = BinaryGibbsNoiseReducer(
        noise_level_prior=experiment.noise_level, observation_strength=1.0, coupling_strength=4.0,
        iterations_count=ITERATIONS_PER_EXPERIMENT, iterations_per_evaluation=ITERATIONS_PER_EVALUATION,
    )
    storage_folder = os.path.join(
        IMAGES_PATH, experiment.experiment_name, f"binary_gibbs_reducer_{round(experiment.noise_level * 100)}"
    )
    run_with_reducer(reducer, experiment.experiment_name, storage_folder)


def run_with_gradient_reducer(experiment):
    reducer = BinaryGradientNoiseReducer(
        noise_level_prior=experiment.noise_level, observation_strength=1.0, coupling_strength=4.0, temperature=2.0,
        iterations_count=ITERATIONS_PER_EXPERIMENT, iterations_per_evaluation=ITERATIONS_PER_EVALUATION,
    )
    storage_folder = os.path.join(
        IMAGES_PATH, experiment.experiment_name, f"binary_gradient_reducer_{round(experiment.noise_level * 100)}"
    )
    run_with_reducer(reducer, experiment.experiment_name, storage_folder)


def run_script():
    for experiment in EXPERIMENTS:
        run_with_gibbs_reducer(experiment)
        print(f"Gibbs sampling for experiment {experiment.experiment_name} done!")
        run_with_gradient_reducer(experiment)
        print(f"Gradient-based sampling for experiment {experiment.experiment_name} done!")


if __name__ == "__main__":
    run_script()
