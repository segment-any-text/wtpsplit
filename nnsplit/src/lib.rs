use std::cmp;
use std::convert::TryInto;
use std::ops::Range;

use ndarray::prelude::*;

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
}

enum SplitInstruction {
    PredictionIndex(usize),
    Function(fn(&str) -> Vec<&str>),
}

struct SplitSequence {
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
    ) -> Split<'a> {
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
                        let part = &text[prev..idx];

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
                        .collect(),
                )),
            }
        } else {
            Split::Text(text)
        }
    }

    pub fn apply<'a>(
        &self,
        text: &'a str,
        predictions: ArrayView2<f32>,
        threshold: f32,
    ) -> Split<'a> {
        self.inner_apply(text, predictions, threshold, 0)
    }
}

pub struct NNSplitOptions {
    threshold: f32,
    stride: usize,
    max_length: usize,
    padding: usize,
}

impl Default for NNSplitOptions {
    fn default() -> Self {
        let max_length = 500;

        NNSplitOptions {
            threshold: 0.1,
            stride: max_length / 2,
            max_length,
            padding: 5,
        }
    }
}

pub trait Backend {
    fn predict(&self, input: Array2<u8>) -> Array3<f32>;
}

#[cfg(feature = "tch-rs-backend")]
pub struct TchRsBackend {
    model: tch::CModule,
    batch_size: usize,
    device: tch::Device,
    n_outputs: usize,
}

#[cfg(feature = "tch-rs-backend")]
impl TchRsBackend {
    pub fn new(model: tch::CModule, device: tch::Device, batch_size: usize) -> Self {
        let dummy_data = tch::Tensor::zeros(&[1, 1], (tch::Kind::Uint8, device));
        let n_outputs = model.forward_ts(&[dummy_data]).unwrap().size()[2] as usize;

        TchRsBackend {
            model,
            device,
            batch_size,
            n_outputs,
        }
    }
}

#[cfg(feature = "tch-rs-backend")]
impl Backend for TchRsBackend {
    fn predict(&self, input: Array2<u8>) -> Array3<f32> {
        let input_shape = input.shape();

        let mut preds = Array3::<f32>::zeros((input_shape[0], input_shape[1], self.n_outputs));

        for i in (0..input_shape[0]).step_by(self.batch_size) {
            let start = i;
            let end = cmp::min(i + self.batch_size, input_shape[0]);

            let batch_inputs = input.slice(s![start..end, ..]).to_slice().unwrap();
            let batch_inputs = tch::Tensor::of_slice(batch_inputs)
                .view((-1, input_shape[1] as i64))
                .to_device(self.device);

            let batch_preds = self.model.forward_ts(&[batch_inputs]).unwrap().sigmoid();
            let batch_preds: ArrayD<f32> = (&batch_preds).try_into().unwrap();

            preds.slice_mut(s![start..end, .., ..]).assign(&batch_preds);
        }

        return preds;
    }
}

pub struct NNSplit {
    backend: Box<dyn Backend>,
    options: NNSplitOptions,
    split_sequence: SplitSequence,
}

impl NNSplit {
    pub fn new(backend: Box<dyn Backend>, options: NNSplitOptions) -> failure::Fallible<NNSplit> {
        Ok(NNSplit {
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
        })
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
            let (pred, count) = preds_and_counts.get_mut(index).unwrap();

            let mut pred_slice = pred.slice_mut(s![range.start..range.end, ..]);
            pred_slice += &slice_pred;

            let mut count_slice = count.slice_mut(s![range.start..range.end, ..]);
            count_slice += 1f32;
        }

        preds_and_counts
            .into_iter()
            .map(|(pred, count): (Array2<f32>, Array2<f32>)| (pred / count))
            .collect()
    }

    pub fn split<'a>(&self, texts: Vec<&'a str>) -> Vec<Split<'a>> {
        let (inputs, indices) = self.get_inputs_and_indeces(&texts);
        let slice_preds = self.backend.predict(inputs);

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
