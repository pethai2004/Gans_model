# Gans_model
Implementation of Generative Model.

Note, however, that using grow model in generative network (Generative_v0) still not work as expected.
There are still problem with multiple keras.optimizer while calling extend_model(). So this need to manually 
save and load model when need to upside (e.g. from 16x16 to 32x32). This will be fixed for my next implementation.
