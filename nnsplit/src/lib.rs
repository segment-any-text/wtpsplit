//! Fast, robust sentence splitting with bindings for Python, Rust and Javascript. This crate contains the core splitting logic which is shared between Javascript, Python and Rust. Each binding then implements a backend separately.
//!
//! See [`tch_rs_backend::NNSplit`](tch_rs_backend/struct.NNSplit.html) for information for using NNSplit from Rust.
#[cfg(test)]
#[macro_use]
extern crate quickcheck_macros;

#[cfg(feature = "model-loader")]
#[macro_use]
extern crate lazy_static;

use ndarray::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::cmp;
use std::ops::Range;

#[cfg(feature = "tch-rs-backend")]
pub mod tch_rs_backend;
#[cfg(feature = "tch-rs-backend")]
pub use tch_rs_backend::NNSplit;

/// Caching and downloading of models.
#[cfg(feature = "model-loader")]
pub mod model_loader;

fn split_whitespace(input: &str) -> Vec<&str> {
    let offset = input.trim_end().len();
    vec![&input[..offset], &input[offset..]]
}

/// A Split level, used to describe what this split corresponds to (e. g. a sentence).
#[derive(Debug, Clone, Copy)]
pub struct Level(&'static str);

#[derive(Debug)]
/// A splitted text.
pub enum Split<'a> {
    /// The lowest level of split.
    Text(&'a str),
    /// A split which contains one or more smaller splits.
    Split((&'a str, Vec<Split<'a>>)),
}

impl<'a> Split<'a> {
    /// Returns the encompassed text.
    pub fn text(&self) -> &'a str {
        match self {
            Split::Split((text, _)) => text,
            Split::Text(text) => text,
        }
    }

    /// Iterate over smaller splits.
    /// # Panics
    /// * If the Split is a `Split::Text` because the lowest level of split can not be iterated over.
    pub fn iter(&self) -> impl Iterator<Item = &Split<'a>> {
        match self {
            Split::Split((_, splits)) => splits.iter(),
            Split::Text(_) => panic!("Can not iterate over Split::Text."),
        }
    }

    /// Recursively flatten the split. Returns a vector where each item is the text of the split at the lowest level.
    pub fn flatten(&self, level: usize) -> Vec<&str> {
        match self {
            Split::Text(text) => vec![text],
            Split::Split((_, parts)) => {
                let mut out = Vec::new();

                for part in parts {
                    if level == 0 {
                        out.push(part.text());
                    } else {
                        out.extend(part.flatten(level - 1));
                    }
                }

                out
            }
        }
    }
}

/// Instruction to split text.
pub enum SplitInstruction {
    /// Instruction to split at the given index of the neural network predictions.
    PredictionIndex(usize),
    /// Instruction to split according to a function.
    Function(fn(&str) -> Vec<&str>),
}

/// Instructions for how to convert neural network outputs and a text to `Split` objects.
pub struct SplitSequence {
    instructions: Vec<(Level, SplitInstruction)>,
}

impl SplitSequence {
    fn new(instructions: Vec<(Level, SplitInstruction)>) -> Self {
        SplitSequence { instructions }
    }

    fn inner_apply<'a>(
        &self,
        text: &'a str,
        predictions: ArrayView2<f32>,
        threshold: f32,
        instruction_idx: usize,
    ) -> Split<'a> {
        assert_eq!(
            predictions.shape()[0],
            text.len(),
            "length of predictions must be equal to the number of bytes in text"
        );

        if let Some((_, instruction)) = self.instructions.get(instruction_idx) {
            match instruction {
                SplitInstruction::PredictionIndex(idx) => {
                    let mut indices: Vec<_> = predictions
                        .slice(s![.., *idx])
                        .indexed_iter()
                        .filter_map(|(index, &item)| {
                            if item > threshold {
                                Some(index + 1)
                            } else {
                                None
                            }
                        })
                        .collect();

                    if indices.is_empty() || indices[indices.len() - 1] != text.len() {
                        indices.push(text.len());
                    }

                    let mut parts = Vec::new();
                    let mut prev = 0;

                    for raw_idx in indices {
                        if prev >= raw_idx {
                            continue;
                        }

                        let mut idx = raw_idx;

                        let part = loop {
                            if let Some(part) = text.get(prev..idx) {
                                break part;
                            }
                            idx += 1;
                        };

                        parts.push(self.inner_apply(
                            part,
                            predictions.slice(s![prev..idx, ..]),
                            threshold,
                            instruction_idx + 1,
                        ));

                        prev = idx;
                    }

                    Split::Split((text, parts))
                }
                SplitInstruction::Function(func) => Split::Split((
                    text,
                    func(text)
                        .iter()
                        .map(|part| {
                            let start = part.as_ptr() as usize - text.as_ptr() as usize;
                            let end = start + part.len();

                            self.inner_apply(
                                part,
                                predictions.slice(s![start..end, ..]),
                                threshold,
                                instruction_idx + 1,
                            )
                        })
                        .collect::<Vec<Split>>(),
                )),
            }
        } else {
            Split::Text(text)
        }
    }

    fn apply<'a>(&self, text: &'a str, predictions: ArrayView2<f32>, threshold: f32) -> Split<'a> {
        self.inner_apply(text, predictions, threshold, 0)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NNSplitOptions {
    #[serde(default = "NNSplitOptions::default_threshold")]
    pub threshold: f32,
    #[serde(default = "NNSplitOptions::default_stride")]
    pub stride: usize,
    #[serde(alias = "maxLength", default = "NNSplitOptions::default_max_length")]
    pub max_length: usize,
    #[serde(default = "NNSplitOptions::default_padding")]
    pub padding: usize,
    #[serde(alias = "batchSize", default = "NNSplitOptions::default_batch_size")]
    pub batch_size: usize,
}

impl NNSplitOptions {
    fn default_threshold() -> f32 {
        0.5
    }

    fn default_stride() -> usize {
        NNSplitOptions::default_max_length() / 2
    }

    fn default_max_length() -> usize {
        500
    }

    fn default_padding() -> usize {
        5
    }

    fn default_batch_size() -> usize {
        256
    }
}

impl Default for NNSplitOptions {
    fn default() -> Self {
        NNSplitOptions {
            threshold: NNSplitOptions::default_threshold(),
            stride: NNSplitOptions::default_stride(),
            max_length: NNSplitOptions::default_max_length(),
            padding: NNSplitOptions::default_padding(),
            batch_size: NNSplitOptions::default_batch_size(),
        }
    }
}

pub struct NNSplitLogic {
    pub options: NNSplitOptions,
    pub split_sequence: SplitSequence,
}

impl NNSplitLogic {
    pub fn new(options: NNSplitOptions) -> Self {
        NNSplitLogic {
            options,
            split_sequence: SplitSequence::new(vec![
                (Level("Sentence"), SplitInstruction::PredictionIndex(0)),
                (Level("Token"), SplitInstruction::PredictionIndex(1)),
                (
                    Level("Whitespace"),
                    SplitInstruction::Function(split_whitespace),
                ),
            ]),
        }
    }

    pub fn get_inputs_and_indices(
        &self,
        texts: &[&str],
    ) -> (Array2<u8>, Vec<(usize, Range<usize>)>) {
        let maxlen = cmp::min(
            texts
                .iter()
                .map(|x| x.len() + self.options.padding * 2)
                .max()
                .unwrap_or(0),
            self.options.max_length,
        );

        let (all_inputs, all_indices) = texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let mut text_inputs: Vec<u8> = Vec::new();
                let mut text_indices: Vec<(usize, Range<usize>)> = Vec::new();

                let length = text.len() + self.options.padding * 2;
                let mut inputs = vec![0; length];

                for (j, byte) in text.bytes().enumerate() {
                    inputs[j + self.options.padding] = byte;
                }

                let mut start = 0;
                let mut end = 0;

                while end != length {
                    end = cmp::min(start + self.options.max_length, length);
                    start = if self.options.max_length > end {
                        0
                    } else {
                        end - self.options.max_length
                    };

                    let mut input_slice = vec![0u8; maxlen];
                    input_slice[..end - start].copy_from_slice(&inputs[start..end]);

                    text_inputs.extend(input_slice);
                    text_indices.push((i, start..end));

                    start += self.options.stride;
                }

                (text_inputs, text_indices)
            })
            .fold(
                (Vec::<u8>::new(), Vec::<(usize, Range<usize>)>::new()),
                |mut acc, (text_inputs, text_indices)| {
                    acc.0.extend(text_inputs);
                    acc.1.extend(text_indices);

                    acc
                },
            );

        let input_array = Array2::from_shape_vec((all_indices.len(), maxlen), all_inputs).unwrap();
        (input_array, all_indices)
    }

    fn combine_predictions(
        &self,
        slice_predictions: ArrayView3<f32>,
        indices: Vec<(usize, Range<usize>)>,
        lengths: Vec<usize>,
    ) -> Vec<Array2<f32>> {
        let pred_dim = slice_predictions.shape()[2];
        let mut preds_and_counts = lengths
            .iter()
            .map(|x| (Array2::zeros((*x, pred_dim)), Array2::zeros((*x, 1))))
            .collect::<Vec<_>>();

        for (slice_pred, (index, range)) in slice_predictions.outer_iter().zip(indices) {
            let (pred, count) = preds_and_counts
                .get_mut(index)
                .expect("slice index must be in bounds");

            let mut pred_slice = pred.slice_mut(s![range.start..range.end, ..]);
            pred_slice += &slice_pred.slice(s![..range.end - range.start, ..]);

            let mut count_slice = count.slice_mut(s![range.start..range.end, ..]);
            count_slice += 1f32;
        }

        preds_and_counts
            .into_iter()
            .map(|(pred, count): (Array2<f32>, Array2<f32>)| (pred / count))
            .collect()
    }

    pub fn split<'a>(
        &self,
        texts: &[&'a str],
        slice_preds: Array3<f32>,
        indices: Vec<(usize, Range<usize>)>,
    ) -> Vec<Split<'a>> {
        let padded_preds = self.combine_predictions(
            (&slice_preds).into(),
            indices,
            texts
                .iter()
                .map(|x| x.len() + self.options.padding * 2)
                .collect(),
        );

        let preds = padded_preds
            .iter()
            .map(|x| {
                x.slice(s![
                    self.options.padding..x.shape()[0] - self.options.padding,
                    ..
                ])
            })
            .collect::<Vec<_>>();

        texts
            .iter()
            .zip(preds)
            .map(|(text, pred)| {
                self.split_sequence
                    .apply(text, pred, self.options.threshold)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    struct DummyNNSplit {
        logic: NNSplitLogic,
    }

    impl DummyNNSplit {
        fn new(options: NNSplitOptions) -> Self {
            DummyNNSplit {
                logic: NNSplitLogic::new(options),
            }
        }

        fn predict(&self, input: Array2<u8>) -> Array3<f32> {
            let n = input.shape()[0];
            let length = input.shape()[1];
            let dim = 2usize;

            let mut rng = thread_rng();

            let mut blob = Vec::new();
            for _ in 0..n * length * dim {
                blob.push(rng.gen_range(0., 1.));
            }

            Array3::from_shape_vec((n, length, dim), blob).unwrap()
        }

        pub fn split<'a>(&self, texts: &[&'a str]) -> Vec<Split<'a>> {
            let (input, indices) = self.logic.get_inputs_and_indices(texts);
            let slice_preds = self.predict(input);

            self.logic.split(texts, slice_preds, indices)
        }
    }

    #[test]
    fn split_instructions_work() {
        let instructions = SplitSequence::new(vec![
            (Level("Token"), SplitInstruction::PredictionIndex(0)),
            (
                Level("Whitespace"),
                SplitInstruction::Function(split_whitespace),
            ),
        ]);

        let input = "This is a test.";
        let mut predictions = array![[0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1.]];
        predictions.swap_axes(0, 1);
        let predictions: ArrayView2<f32> = (&predictions).into();

        let splits = instructions.apply(input, predictions, 0.5);
        assert_eq!(splits.flatten(0), ["This ", "is ", "a ", "test", "."]);
        assert_eq!(
            splits.flatten(1),
            ["This", " ", "is", " ", "a", " ", "test", "", ".", ""]
        );
    }

    #[test]
    fn splitter_works() {
        let options = NNSplitOptions {
            stride: 5,
            max_length: 20,
            ..NNSplitOptions::default()
        };
        let splitter = DummyNNSplit::new(options);

        // sample text must only contain chars which are 1 byte long, so that `DummyNNSplit`
        // can not generate splits which are not char boundaries
        splitter.split(&["This is a short test.", "This is another short test."]);
    }

    #[test]
    fn splitter_works_on_empty_input() {
        let splitter = DummyNNSplit::new(NNSplitOptions::default());

        let splits = splitter.split(&[]);
        assert!(splits.is_empty());
    }

    #[quickcheck]
    fn length_invariant(text: String) -> bool {
        let splitter = DummyNNSplit::new(NNSplitOptions::default());

        let split = &splitter.split(&[&text])[0];

        let mut sums: Vec<usize> = Vec::new();
        sums.push(split.iter().map(|x| x.text().len()).sum());

        for i in 0..4 {
            sums.push(split.flatten(i).iter().map(|x| x.len()).sum());
        }

        sums.into_iter().all(|sum| sum == text.len())
    }
}
