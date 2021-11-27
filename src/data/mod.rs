mod reader;

pub use crate::data::reader::CsvReader;
use rand::prelude::*;
use std::collections::HashMap;

pub struct Dataset {
    pub user_to_idx: HashMap<String, usize>,
    pub item_to_idx: HashMap<String, usize>,
    pub users: Vec<String>,
    pub items: Vec<String>,
    pub values: Vec<f64>,
}

impl Dataset {
    pub fn new<T>(data: T) -> Self
    where
        T: Iterator<Item = (String, String, f64)>,
    {
        let mut user_to_idx = HashMap::new();
        let mut item_to_idx = HashMap::new();
        let mut users = vec![];
        let mut items = vec![];
        let mut values = vec![];

        for (user, item, value) in data {
            if !user_to_idx.contains_key(&user) {
                user_to_idx.insert(user.clone(), user_to_idx.len());
            }

            if !item_to_idx.contains_key(&item) {
                item_to_idx.insert(item.clone(), item_to_idx.len());
            }

            users.push(user);
            items.push(item);
            values.push(value);
        }

        Self {
            user_to_idx,
            item_to_idx,
            users,
            items,
            values,
        }
    }

    pub fn len(&self) -> usize {
        self.users.len()
    }

    pub fn shuffle(self) -> Self {
        self.shuffle_with(thread_rng())
    }

    pub fn shuffle_with(mut self, mut rng: ThreadRng) -> Self {
        for i in 0..self.users.len() {
            let j = rng.gen_range(0..=i);
            self.users.swap(i, j);
            self.items.swap(i, j);
            self.values.swap(i, j);
        }

        self
    }

    pub fn train_test_split(mut self, num_train: usize) -> (Self, Self) {
        let testset = Dataset {
            user_to_idx: self.user_to_idx.clone(),
            item_to_idx: self.item_to_idx.clone(),
            users: self.users.split_off(num_train),
            items: self.items.split_off(num_train),
            values: self.values.split_off(num_train),
        };

        let trainset = Dataset {
            user_to_idx: self.user_to_idx,
            item_to_idx: self.item_to_idx,
            users: self.users,
            items: self.items,
            values: self.values,
        };

        (trainset, testset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_test_split() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let (trainset, testset) = Dataset::new(csv_reader.into_iter()).train_test_split(60);
        println!("{}, {}", trainset.len(), testset.len());
    }

    #[test]
    fn test_shuffle() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let dataset = Dataset::new(csv_reader.into_iter()).shuffle();
    }
}
