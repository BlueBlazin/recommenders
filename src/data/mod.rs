mod reader;

pub use crate::data::reader::CsvReader;
use rand::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Dataset<'a> {
    pub user_to_idx: Rc<HashMap<String, usize>>,
    pub item_to_idx: Rc<HashMap<String, usize>>,
    pub users: Cow<'a, [String]>,
    pub items: Cow<'a, [String]>,
    pub values: Cow<'a, [f64]>,
}

impl<'a> Dataset<'a> {
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
            user_to_idx: Rc::from(user_to_idx),
            item_to_idx: Rc::from(item_to_idx),
            users: Cow::from(users),
            items: Cow::from(items),
            values: Cow::from(values),
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
            self.users.to_mut().swap(i, j);
            self.items.to_mut().swap(i, j);
            self.values.to_mut().swap(i, j);
        }

        self
    }

    pub fn train_test_split(&'a self, num_train: usize) -> (Self, Self) {
        let trainset = Dataset {
            user_to_idx: Rc::clone(&self.user_to_idx),
            item_to_idx: Rc::clone(&self.item_to_idx),
            users: Cow::from(&self.users[..num_train]),
            items: Cow::from(&self.items[..num_train]),
            values: Cow::from(&self.values[..num_train]),
        };

        let testset = Dataset {
            user_to_idx: Rc::clone(&self.user_to_idx),
            item_to_idx: Rc::clone(&self.item_to_idx),
            users: Cow::from(&self.users[num_train..]),
            items: Cow::from(&self.items[num_train..]),
            values: Cow::from(&self.values[num_train..]),
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
        let dataset = Dataset::new(csv_reader.into_iter());
        let (trainset, testset) = dataset.train_test_split(60);
        println!("{}, {}", trainset.len(), testset.len());
    }

    #[test]
    fn test_shuffle() {
        let csv_reader = CsvReader::new("./test.csv", (0, 1, 2), b'\t', false);
        let dataset = Dataset::new(csv_reader.into_iter()).shuffle();
    }
}
