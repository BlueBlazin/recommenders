extern crate nalgebra as na;

pub mod algos;
pub mod data;
pub mod objectives;

/*
Desired features:
1. allow training with different loss functions including custom ones
2. first class support for csv files, easy to use normal csvs (not wrapped) to train
3. allow transforms and inverse transform data including custom transforms
4. first class support to query internals such as learned embeddings, mapping
   values to internal integer indices
5. simple API for integrating in other programs
6. a plug and play command line interface
7. speed
*/
