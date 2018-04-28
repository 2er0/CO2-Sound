# Test & Train - Data

`/data/wav` contains the training and test data

* `/data/wav/Train`: raw training data with labels (wav/audio)
* `/data/wav/Test`: raw test data without labels (wav/audio)
* `/data/wav/Bin`: numpy.ndarray in bin format (parsed wav's)

# npy-generation

`.npy` files can be generated from the wav-source via `converter.py`