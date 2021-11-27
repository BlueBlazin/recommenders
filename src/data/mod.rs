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
}

pub struct CsvReader {
    data: Vec<(String, String, f64)>,
}

impl CsvReader {
    pub fn new(path: &str, cols: (usize, usize, usize), delimiter: u8, has_headers: bool) -> Self {
        let (user_col, item_col, value_col) = cols;

        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(has_headers)
            .from_path(path)
            .unwrap();

        let mut data = vec![];

        for result in rdr.records() {
            let record = result.unwrap();
            let user = record.get(user_col).unwrap().to_owned();
            let item = record.get(item_col).unwrap().to_owned();
            let value: f64 = record.get(value_col).unwrap().parse().unwrap();
            data.push((user, item, value));
        }

        Self { data }
    }
}

impl IntoIterator for CsvReader {
    type Item = (String, String, f64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}
