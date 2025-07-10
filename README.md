# Testing ML Models on Graph Games

To run this ML model on the GAMMS simulator, three additional python packages are needed: `torch`, `lightning`, and `torch_geometric`. These three packages are used in this code to build and train the GNN model. Currently, the model is just an MLP, and the code can be simplified to only require `torch`. Once the GNNs (and maybe transformers) are implemented, the other two libraries become helpful to simplify the process.

The test model can be found in the `models/` folder, while the strategy code is in `strategy.py`. The code can be run as the league code would normally be run; the `models/` folder just needs to be in the same directory as `strategy.py` (namely, the `policies/` directory). On your machine, you can just copy the `strategy.py` file and `models/` folder into the `policies/` directory.

Keep in mind that this model likely does not work very well, it is just a way to test the compatibility of these libraries with the League code. Please let me know if you have any questions.
