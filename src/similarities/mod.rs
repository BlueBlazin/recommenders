mod cosine;
mod msd;

pub use crate::similarities::cosine::cosine;
pub use crate::similarities::msd::msd;

pub enum Similarity {
    Cosine,
    Msd,
}
