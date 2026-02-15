from dataset.sources.devemo_source import DevemoSource
from dataset.sources.fer2013_source import FER2013Source
from dataset.sources.veatic_source import VEATICSource


def get_dataset_source(input_flag, input_dir):
    if input_flag == "fer2013":
        return FER2013Source(input_dir)
    if input_flag == "devemo":
        return DevemoSource(input_dir, plus_variant=False)
    if input_flag == "devemo+":
        return DevemoSource(input_dir, plus_variant=True)
    if input_flag == "veatic":
        return VEATICSource(input_dir)

    raise ValueError(f"Unsupported input dataset: {input_flag}")
