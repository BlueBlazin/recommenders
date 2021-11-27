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
