#[cfg(test)]
#[macro_use]
extern crate quickcheck_macros;

#[macro_use]
extern crate lazy_static;

use ndarray::prelude::*;
use std::cmp;
use std::error::Error;
use std::ops::Range;
use thiserror::Error;

#[cfg(feature = "tch-rs-backend")]
mod tch_rs_backend;

#[cfg(feature = "model-loader")]
mod model_loader;

fn split_whitespace(input: &str) -> Vec<&str> {
    let offset = input.trim_end().len();
    vec![&input[..offset], &input[offset..]]
}

#[derive(Debug, Clone, Copy)]
pub struct Level(&'static str);

#[derive(Debug)]
pub enum Split<'a> {
    Text(&'a str),
    Split((&'a str, Vec<Split<'a>>)),
}

impl<'a> Split<'a> {
    pub fn text(&self) -> &'a str {
        match self {
            Split::Split((text, _)) => text,
            Split::Text(text) => text,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Split<'a>> {
        match self {
            Split::Split((_, splits)) => splits.iter(),
            Split::Text(_) => panic!("Can not iterate over Split::Text."),
        }
    }

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

pub enum SplitInstruction {
    PredictionIndex(usize),
    Function(fn(&str) -> Vec<&str>),
}

#[derive(Error, Debug)]
pub enum SplitCreationError {
    #[error("indexing {text} from {start} to {end} failed (possible char boundary error)")]
    IndexError {
        text: String,
        start: usize,
        end: usize,
    },
}

pub struct SplitSequence {
    instructions: Vec<(Level, SplitInstruction)>,
}

impl SplitSequence {
    pub fn new(instructions: Vec<(Level, SplitInstruction)>) -> Self {
        SplitSequence { instructions }
    }

    fn inner_apply<'a>(
        &self,
        text: &'a str,
        predictions: ArrayView2<f32>,
        threshold: f32,
        instruction_idx: usize,
    ) -> Result<Split<'a>, SplitCreationError> {
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

                    if indices.len() == 0 || indices[indices.len() - 1] != text.len() {
                        indices.push(text.len());
                    }

                    let mut parts = Vec::new();
                    let mut prev = 0;

                    for idx in indices {
                        let part =
                            text.get(prev..idx)
                                .ok_or_else(|| SplitCreationError::IndexError {
                                    text: text.to_owned(),
                                    start: prev,
                                    end: idx,
                                })?;

                        parts.push(self.inner_apply(
                            part,
                            predictions.slice(s![prev..idx, ..]),
                            threshold,
                            instruction_idx + 1,
                        )?);

                        prev = idx;
                    }

                    Ok(Split::Split((text, parts)))
                }
                SplitInstruction::Function(func) => Ok(Split::Split((
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
                        .collect::<Result<Vec<Split>, SplitCreationError>>()?,
                ))),
            }
        } else {
            Ok(Split::Text(text))
        }
    }

    pub fn apply<'a>(
        &self,
        text: &'a str,
        predictions: ArrayView2<f32>,
        threshold: f32,
    ) -> Result<Split<'a>, SplitCreationError> {
        self.inner_apply(text, predictions, threshold, 0)
    }
}

#[derive(Error, Debug)]
pub enum SplitError {
    #[error(transparent)]
    CreationError { source: SplitCreationError },
    #[error(transparent)]
    BackendError { source: Box<dyn Error> },
}

pub struct NNSplitOptions {
    pub threshold: f32,
    pub stride: usize,
    pub max_length: usize,
    pub padding: usize,
    pub batch_size: usize,
}

impl Default for NNSplitOptions {
    fn default() -> Self {
        let max_length = 500;

        NNSplitOptions {
            threshold: 0.1,
            stride: max_length / 2,
            max_length,
            padding: 5,
            batch_size: 128,
        }
    }
}

pub trait Backend {
    fn predict(&self, input: Array2<u8>, batch_size: usize) -> Result<Array3<f32>, Box<dyn Error>>;
}

pub struct NNSplit {
    backend: Box<dyn Backend>,
    options: NNSplitOptions,
    split_sequence: SplitSequence,
}

impl NNSplit {
    #[cfg(feature = "tch-rs-backend")]
    pub fn new<P: AsRef<std::path::Path>>(
        model_path: P,
        device: tch::Device,
        options: NNSplitOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let model = tch::CModule::load(model_path)?;
        let backend = tch_rs_backend::TchRsBackend::new(model, device);

        Ok(Self::from_backend(Box::new(backend), options))
    }

    #[cfg(all(feature = "tch-rs-backend", feature = "model-loader"))]
    pub fn load(
        model_name: &str,
        device: tch::Device,
        options: NNSplitOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let filename = match device {
            tch::Device::Cpu => "torchscript_cpu_model.pt",
            tch::Device::Cuda(_) => "torchscript_cuda_model.pt",
        };
        let mut model_data = model_loader::get_from_cache_or_download(model_name, filename)?;
        let model = tch::CModule::load_data(&mut model_data)?;
        let backend = tch_rs_backend::TchRsBackend::new(model, device);

        Ok(Self::from_backend(Box::new(backend), options))
    }

    pub fn from_backend(backend: Box<dyn Backend>, options: NNSplitOptions) -> Self {
        NNSplit {
            backend,
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

    fn get_inputs_and_indeces(
        &self,
        texts: &Vec<&str>,
    ) -> (Array2<u8>, Vec<(usize, Range<usize>)>) {
        let maxlen = cmp::min(
            texts
                .iter()
                .map(|x| x.len() + self.options.padding * 2)
                .max()
                .unwrap_or(0),
            self.options.max_length,
        );

        let mut all_inputs: Vec<u8> = Vec::new();
        let mut all_indices: Vec<(usize, Range<usize>)> = Vec::new();

        for (i, text) in texts.iter().enumerate() {
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

                all_inputs.extend(input_slice);
                all_indices.push((i, start..end));

                start += self.options.stride;
            }
        }

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

    pub fn split<'a>(&self, texts: Vec<&'a str>) -> Result<Vec<Split<'a>>, SplitError> {
        let (inputs, indices) = self.get_inputs_and_indeces(&texts);
        let slice_preds = self
            .backend
            .predict(inputs, self.options.batch_size)
            .map_err(|source| SplitError::BackendError { source })?;

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
                    .map_err(|source| SplitError::CreationError { source })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    struct DummyBackend {}

    impl Backend for DummyBackend {
        fn predict(
            &self,
            input: Array2<u8>,
            _batch_size: usize,
        ) -> Result<Array3<f32>, Box<dyn Error>> {
            let n = input.shape()[0];
            let length = input.shape()[1];
            let dim = 2usize;

            let mut rng = thread_rng();

            let mut blob = Vec::new();
            for _ in 0..n * length * dim {
                blob.push(rng.gen_range(0., 1.));
            }

            Ok(Array3::from_shape_vec((n, length, dim), blob)?)
        }
    }

    fn dummy_nnsplit(options: NNSplitOptions) -> NNSplit {
        NNSplit::from_backend(Box::new(DummyBackend {}), options)
    }

    #[test]
    fn split_instructions_work() -> Result<(), SplitCreationError> {
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

        let splits = instructions.apply(input, predictions, 0.5)?;
        assert_eq!(splits.flatten(0), ["This ", "is ", "a ", "test", "."]);
        assert_eq!(
            splits.flatten(1),
            ["This", " ", "is", " ", "a", " ", "test", "", ".", ""]
        );

        Ok(())
    }

    #[cfg(all(feature = "tch-rs-backend", feature = "model-loader"))]
    #[test]
    fn splitter_model_works() -> Result<(), Box<dyn Error>> {
        let splitter = NNSplit::load("de", tch::Device::Cpu, NNSplitOptions::default())?;
        let splits = &splitter.split(vec!["Das ist ein Test Das ist noch ein Test."])?[0];

        assert_eq!(
            splits.flatten(0),
            vec!["Das ist ein Test ", "Das ist noch ein Test."]
        );
        Ok(())
    }

    #[test]
    fn splitter_works() -> Result<(), SplitError> {
        let options = NNSplitOptions {
            stride: 5,
            max_length: 20,
            ..NNSplitOptions::default()
        };
        let splitter = dummy_nnsplit(options);

        // sample text must only contain chars which are 1 byte long, so that `DummyBackend`
        // can not generate splits which are not char boundaries
        splitter.split(vec!["This is a short test.", "This is another short test."])?;
        Ok(())
    }

    #[test]
    fn splitter_works_on_empty_input() -> Result<(), SplitError> {
        let splitter = dummy_nnsplit(NNSplitOptions::default());

        let splits = splitter.split(vec![])?;
        assert!(splits.len() == 0);
        Ok(())
    }

    #[quickcheck]
    fn length_invariant(text: String) -> bool {
        let splitter = dummy_nnsplit(NNSplitOptions::default());

        let splits_result = splitter.split(vec![&text]);

        if let Ok(splits) = splits_result {
            let split = &splits[0];

            let mut sums: Vec<usize> = Vec::new();

            sums.push(split.iter().map(|x| x.text().len()).sum());

            for i in 0..4 {
                sums.push(split.flatten(i).iter().map(|x| x.len()).sum());
            }

            sums.into_iter().all(|sum| sum == text.len())
        } else {
            true
        }
    }
}
