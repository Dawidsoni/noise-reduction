import os
import numpy as np
import pickle
from dataclasses import dataclass
import itertools
from multiprocessing import Pool
import PIL

from noise_reducers.grayscale_gibbs_noise_reducer import GrayscaleGibbsNoiseReducer
from noise_reducers.grayscale_gradient_noise_reducer import GrayscaleGradientNoiseReducer
from noise_reducers import image_utils


@dataclass
class Experiment(object):
    experiment_name: str
    noise_level: float


EXPERIMENTS = [
    Experiment(experiment_name="size800_noise10_flipped", noise_level=0.1),
    Experiment(experiment_name="size800_noise20_flipped", noise_level=0.2),
]
IMAGES_PATH = "../grayscale_images"
IMAGES_PER_EXPERIMENT = 20
ITERATIONS_PER_EXPERIMENT = 300_000
ITERATIONS_PER_EVALUATION = 6000


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
            ground_truth=image_utils.load_grayscale_image_as_numpy_array(ground_truth_path),
            observation=image_utils.load_grayscale_image_as_numpy_array(observation_path),
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
    print(f"Gibbs sampling for experiment {experiment.experiment_name} started!")
    reducer = GrayscaleGibbsNoiseReducer(
        noise_level_prior=experiment.noise_level, observation_strength=1.0, coupling_strength=4.0,
        iterations_count=ITERATIONS_PER_EXPERIMENT, iterations_per_evaluation=ITERATIONS_PER_EVALUATION,
    )
    storage_folder = os.path.join(
        IMAGES_PATH, experiment.experiment_name, f"grayscale_gibbs_reducer_{round(experiment.noise_level * 100)}"
    )
    run_with_reducer(reducer, experiment.experiment_name, storage_folder)
    print(f"Gibbs sampling for experiment {experiment.experiment_name} done!")


def run_with_gradient_reducer(experiment):
    print(f"Gradient-based sampling for experiment {experiment.experiment_name} started!")
    reducer = GrayscaleGradientNoiseReducer(
        noise_level_prior=experiment.noise_level, observation_strength=1.0, coupling_strength=4.0, temperature=2.0,
        iterations_count=ITERATIONS_PER_EXPERIMENT, iterations_per_evaluation=ITERATIONS_PER_EVALUATION,
    )
    storage_folder = os.path.join(
        IMAGES_PATH, experiment.experiment_name, f"grayscale_gradient_reducer_{round(experiment.noise_level * 100)}"
    )
    run_with_reducer(reducer, experiment.experiment_name, storage_folder)
    print(f"Gradient-based sampling for experiment {experiment.experiment_name} done!")


def run_with_reducer_type(experiment, reducer_type):
    if reducer_type == "gibbs":
        run_with_gibbs_reducer(experiment)
    elif reducer_type == "gradient":
        run_with_gradient_reducer(experiment)
    else:
        raise ValueError("Invalid type of reducer")


def run_script():
    arguments_to_run = list(itertools.chain(
        zip(EXPERIMENTS, ["gibbs"] * len(EXPERIMENTS)),
        zip(EXPERIMENTS, ["gradient"] * len(EXPERIMENTS)),
    ))
    with Pool(processes=6) as pool:
        pool.starmap(run_with_reducer_type, arguments_to_run)


if __name__ == "__main__":
    run_script()
