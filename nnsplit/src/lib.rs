//! Fast, robust sentence splitting with bindings for Python, Rust and Javascript. This crate contains the core splitting logic which is shared between Javascript, Python and Rust. Each binding then implements a backend separately.
//!
//! See [`tract_backend::NNSplit`](tract_backend/struct.NNSplit.html) for information for using NNSplit from Rust.
#![warn(missing_docs)]
#[cfg(test)]
#[macro_use]
extern crate quickcheck_macros;

use lazy_static::lazy_static;
use ndarray::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::cmp;
use std::collections::HashMap;
use std::ops::Range;

/// Backend to run models using tch-rs.
#[cfg(feature = "tract-backend")]
pub mod tract_backend;
#[cfg(feature = "tract-backend")]
pub use tract_backend::NNSplit;

/// Caching and downloading of models.
#[cfg(feature = "model-loader")]
pub mod model_loader;

/// A Split level, used to describe what this split corresponds to (e. g. a sentence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level(String);

/// A splitted text.
#[derive(Debug)]
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

fn split_whitespace(input: &str) -> Vec<&str> {
    let offset = input.trim_end().len();
    vec![&input[..offset], &input[offset..]]
}

type SplitFunction = fn(&str) -> Vec<&str>;

lazy_static! {
    static ref SPLIT_FUNCTIONS: HashMap<&'static str, SplitFunction> = {
        let mut map = HashMap::new();
        map.insert("whitespace", split_whitespace as SplitFunction);
        map
    };
}

#[derive(Serialize, Deserialize)]
/// Instruction to split text.
pub enum SplitInstruction {
    /// Instruction to split at the given index of the neural network predictions.
    PredictionIndex(usize),
    /// Instruction to split according to a function.
    Function(String),
}

#[derive(Serialize, Deserialize)]
/// Instructions for how to convert neural network outputs and a text to `Split` objects.
pub struct SplitSequence {
    instructions: Vec<(Level, SplitInstruction)>,
}

impl SplitSequence {
    /// Creates a new split sequence. Contains instructions for how to use model predictions to split a text.
    pub fn new(instructions: Vec<(Level, SplitInstruction)>) -> Self {
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
                SplitInstruction::Function(func_name) => Split::Split((
                    text,
                    (*SPLIT_FUNCTIONS.get(func_name.as_str()).unwrap())(text)
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

/// Options for splitting text.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NNSplitOptions {
    /// Threshold from 0 to 1 above which predictions will be considered positive.
    #[serde(default = "NNSplitOptions::default_threshold")]
    pub threshold: f32,
    /// How much to move the window after each prediction (comparable to stride of 1d convolution).
    #[serde(default = "NNSplitOptions::default_stride")]
    pub stride: usize,
    /// The maximum length of each cut (comparable to kernel size of 1d convolution).
    #[serde(alias = "maxLength", default = "NNSplitOptions::default_max_length")]
    pub max_length: usize,
    /// How much to zero pad the text on both sides.
    #[serde(default = "NNSplitOptions::default_padding")]
    pub padding: usize,
    /// Total length will be padded until it is divisible by this number. Allows some additional optimizations.
    #[serde(
        alias = "paddingDivisor",
        default = "NNSplitOptions::default_length_divisor"
    )]
    pub length_divisor: usize,
    /// Batch size to use.
    #[serde(alias = "batchSize", default = "NNSplitOptions::default_batch_size")]
    pub batch_size: usize,
}

impl NNSplitOptions {
    fn default_threshold() -> f32 {
        0.8
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

    fn default_length_divisor() -> usize {
        2
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
            length_divisor: NNSplitOptions::default_length_divisor(),
        }
    }
}

/// The logic by which texts are split.
pub struct NNSplitLogic {
    #[allow(missing_docs)]
    pub options: NNSplitOptions,
    #[allow(missing_docs)]
    pub split_sequence: SplitSequence,
}

impl NNSplitLogic {
    /// Create new logic from options. The split sequence is not customizable at the moment.
    ///
    /// # Panics
    /// - If the options are invalid, e. g. max_length % length_divisor != 0
    pub fn new(options: NNSplitOptions, split_sequence: SplitSequence) -> Self {
        if options.max_length % options.length_divisor != 0 {
            panic!("max length must be divisible by length divisor.")
        }

        NNSplitLogic {
            options,
            split_sequence,
        }
    }

    fn pad(&self, length: usize) -> usize {
        let padded = length + self.options.padding * 2;
        let remainder = padded % self.options.length_divisor;

        if remainder == 0 {
            padded
        } else {
            padded + (self.options.length_divisor - remainder)
        }
    }

    /// Convert texts to neural network inputs. Returns:
    /// * An `ndarray::Array2` which can be fed into the neural network as is.
    /// * A vector of indices with information which positions in the text the array elements correspond to.
    pub fn get_inputs_and_indices(
        &self,
        texts: &[&str],
    ) -> (Array2<u8>, Vec<(usize, Range<usize>)>) {
        let maxlen = cmp::min(
            texts.iter().map(|x| self.pad(x.len())).max().unwrap_or(0),
            self.options.max_length,
        );

        let (all_inputs, all_indices) = texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let mut text_inputs: Vec<u8> = Vec::new();
                let mut text_indices: Vec<(usize, Range<usize>)> = Vec::new();

                let length = self.pad(text.len());
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

    /// Splits the text, given predictions by a neural network and indices
    /// with information which positions in the text the predictions correspond to.
    pub fn split<'a>(
        &self,
        texts: &[&'a str],
        slice_preds: Array3<f32>,
        indices: Vec<(usize, Range<usize>)>,
    ) -> Vec<Split<'a>> {
        let padded_preds = self.combine_predictions(
            (&slice_preds).into(),
            indices,
            texts.iter().map(|x| self.pad(x.len())).collect(),
        );

        let preds = padded_preds
            .iter()
            .zip(texts)
            .map(|(x, text)| {
                x.slice(s![
                    self.options.padding..self.options.padding + text.len(),
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
                logic: NNSplitLogic::new(
                    options,
                    SplitSequence::new(vec![
                        (
                            Level("Sentence".into()),
                            SplitInstruction::PredictionIndex(0),
                        ),
                        (Level("Token".into()), SplitInstruction::PredictionIndex(1)),
                        (
                            Level("Whitespace".into()),
                            SplitInstruction::Function("whitespace".into()),
                        ),
                    ]),
                ),
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
            (Level("Token".into()), SplitInstruction::PredictionIndex(0)),
            (
                Level("Whitespace".into()),
                SplitInstruction::Function("whitespace".into()),
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
