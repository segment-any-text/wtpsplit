use std::cmp;
use std::convert::{TryFrom, TryInto};
use std::io::Cursor;
use std::ops::Range;
use std::vec::Vec;

use ndarray::prelude::*;
use ndarray::Array2;
use tch::{Device, Tensor};
use unicode_segmentation::UnicodeSegmentation;

fn text_to_id(character: &str) -> u8 {
    if character.chars().count() != 1 {
        return 1;
    }
    let ord = character.chars().next().unwrap() as u32;
    if ord <= 127 {
        u8::try_from(ord + 2).unwrap()
    } else {
        1
    }
}

#[derive(Debug, PartialEq)]
pub struct Token {
    pub text: String,
    pub whitespace: String,
}

impl Token {
    fn new(text: String) -> Token {
        let last_char_index = match text.chars().rev().position(|x| !x.is_whitespace()) {
            Some(x) => x,
            None => 0,
        };
        let last_char_index = text.len() - last_char_index;

        Token {
            text: text[..last_char_index].to_string(),
            whitespace: text[last_char_index..].to_string(),
        }
    }
}

pub struct NNSplit {
    model: tch::CModule,
    threshold: f32,
    stride: usize,
    cut_length: usize,
    batch_size: usize,
    padding: usize,
    device: Device,
}

static DE_DATA_CPU: &'static [u8] = include_bytes!("../data/de/ts_cpu.pt");
static DE_DATA_CUDA: &'static [u8] = include_bytes!("../data/de/ts_cuda.pt");

static EN_DATA_CPU: &'static [u8] = include_bytes!("../data/en/ts_cpu.pt");
static EN_DATA_CUDA: &'static [u8] = include_bytes!("../data/en/ts_cuda.pt");

impl NNSplit {
    const THRESHOLD: f32 = 0.1;
    const CUT_LENGTH: usize = 500;
    const STRIDE: usize = NNSplit::CUT_LENGTH / 2;
    const BATCH_SIZE: usize = 32;
    const PADDING: usize = 5;

    /// Returns an An NNSplit sentencizer and tokenizer.
    ///
    /// # Arguments
    ///
    /// * `model_name` - Name of the prepackaged model to use. Either `en` or `de`.
    pub fn new(model_name: &str) -> failure::Fallible<NNSplit> {
        let device = Device::cuda_if_available();
        let source = if model_name == "de" {
            if device == Device::Cpu {
                DE_DATA_CPU
            } else {
                DE_DATA_CUDA
            }
        } else if model_name == "en" {
            if device == Device::Cpu {
                EN_DATA_CPU
            } else {
                EN_DATA_CUDA
            }
        } else {
            panic!(format!("unknown model name: {}.", model_name));
        };

        let mut cursor = Cursor::new(source);
        let model = tch::CModule::load_data(&mut cursor)?;

        Ok(NNSplit {
            model,
            threshold: NNSplit::THRESHOLD,
            stride: NNSplit::STRIDE,
            cut_length: NNSplit::CUT_LENGTH,
            batch_size: NNSplit::BATCH_SIZE,
            padding: NNSplit::PADDING,
            device,
        })
    }

    /// Returns an An NNSplit sentencizer and tokenizer.
    ///
    /// # Arguments
    ///
    /// * `model` - tch::CModule that will be used as the model.
    pub fn from_model(model: tch::CModule) -> failure::Fallible<NNSplit> {
        let device = Device::cuda_if_available();

        Ok(NNSplit {
            model,
            threshold: NNSplit::THRESHOLD,
            stride: NNSplit::STRIDE,
            cut_length: NNSplit::CUT_LENGTH,
            batch_size: NNSplit::BATCH_SIZE,
            padding: NNSplit::PADDING,
            device,
        })
    }

    pub fn with_threshold(&mut self, threshold: f32) -> &mut Self {
        self.threshold = threshold;
        self
    }

    pub fn with_stride(&mut self, stride: usize) -> &mut Self {
        self.stride = stride;
        self
    }

    pub fn with_cut_length(&mut self, cut_length: usize) -> &mut Self {
        self.cut_length = cut_length;
        self
    }

    pub fn with_batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_device(&mut self, device: Device) -> &mut Self {
        self.device = device;
        self
    }

    fn get_raw_preds(
        &self,
        texts: &Vec<&str>,
        text_lengths: &Vec<usize>,
    ) -> (Array3<f32>, Vec<Range<usize>>, Vec<usize>) {
        let mut all_inputs: Vec<u8> = Vec::new();
        let mut all_idx: Vec<Range<usize>> = Vec::new();
        let mut n_cuts_per_text: Vec<usize> = Vec::new();

        for (text, length) in texts.iter().zip(text_lengths.iter()) {
            let current_length = cmp::max(*length, self.cut_length) + self.padding * 2;
            let mut inputs = vec![0; current_length];
            for (i, character) in text.graphemes(true).enumerate() {
                inputs[i + self.padding] = text_to_id(character);
            }

            let mut start = 0;
            let mut end = 0;
            let mut i: usize = 0;

            while end != current_length {
                end = cmp::min(start + self.cut_length, current_length);
                start = end - self.cut_length;

                let range = Range { start, end };
                all_inputs.extend(inputs[range.clone()].iter());
                all_idx.push(range);
                start += self.stride;
                i += 1;
            }

            n_cuts_per_text.push(i);
        }

        let mut preds = Array3::<f32>::zeros((all_idx.len(), self.cut_length, 2));
        for i in (0..all_idx.len()).step_by(self.batch_size) {
            let batch_start = i;
            let batch_end = cmp::min(i + self.batch_size, all_idx.len());

            let range = Range {
                start: batch_start * self.cut_length,
                end: batch_end * self.cut_length,
            };
            let batch_inputs = Tensor::of_slice(&all_inputs[range])
                .view((-1, self.cut_length as i64))
                .to_device(self.device);
            let batch_preds = self.model.forward_ts(&[batch_inputs]).unwrap().sigmoid();
            let batch_preds: ArrayD<f32> = (&batch_preds).try_into().unwrap();

            preds
                .slice_mut(s![batch_start..batch_end, .., ..])
                .assign(&batch_preds);
        }

        return (preds, all_idx, n_cuts_per_text);
    }

    fn average_preds(
        &self,
        preds: Array3<f32>,
        all_idx: Vec<Range<usize>>,
        n_cuts_per_text: Vec<usize>,
        text_lengths: &Vec<usize>,
    ) -> Vec<Array2<f32>> {
        let mut all_avg_preds = text_lengths
            .iter()
            .map(|x| Array2::<f32>::zeros((*x, 2)))
            .collect::<Vec<_>>();
        let mut all_avg_pred_counts = text_lengths
            .iter()
            .map(|x| Array2::<f32>::zeros((*x, 1)))
            .collect::<Vec<_>>();

        let mut current_text = 0;
        let mut current_i = 0;

        for i in 0..n_cuts_per_text.iter().sum() {
            let current_preds = &mut all_avg_preds[current_text];
            let current_pred_counts = &mut all_avg_pred_counts[current_text];

            let cutoff = if all_idx[i].start < self.padding {
                self.padding - all_idx[i].start
            } else {
                0
            };
            let range = Range {
                start: if cutoff == 0 {
                    all_idx[i].start - self.padding
                } else {
                    0
                },
                end: cmp::min(all_idx[i].end - self.padding, text_lengths[current_text]),
            };
            let mut slice = current_preds.slice_mut(s![range.clone(), ..]);
            slice += &preds.slice(s![i, cutoff..range.end - range.start + cutoff, ..]);

            let mut n_slice = current_pred_counts.slice_mut(s![range, 0]);
            n_slice += 1.;

            current_i += 1;
            if current_i == n_cuts_per_text[current_text] {
                let mut sum_of_preds = current_preds.slice_mut(s![.., ..]);
                sum_of_preds /= &current_pred_counts.slice(s![.., ..]);

                current_text += 1;
                current_i = 0;
            }
        }

        all_avg_preds
    }

    fn split_texts_from_preds(
        &self,
        texts: Vec<&str>,
        all_avg_preds: Vec<Array2<f32>>,
    ) -> Vec<Vec<Vec<Token>>> {
        let mut tokenized_texts: Vec<Vec<Vec<Token>>> = Vec::new();
        for (avg_preds, text) in all_avg_preds.iter().zip(texts.iter()) {
            let mut sentences: Vec<Vec<Token>> = Vec::new();
            let mut tokens: Vec<Token> = Vec::new();
            let mut token = String::new();

            for (i, character) in text.graphemes(true).enumerate() {
                token += character;
                // also consider sentence end as token end
                if avg_preds[[i, 0]] > self.threshold || avg_preds[[i, 1]] > self.threshold {
                    tokens.push(Token::new(token));
                    token = String::new();
                }

                if avg_preds[[i, 1]] > self.threshold {
                    sentences.push(tokens);
                    tokens = Vec::new();
                }
            }

            if token.len() > 0 {
                tokens.push(Token::new(token));
            }

            if tokens.len() > 0 {
                sentences.push(tokens);
            }

            tokenized_texts.push(sentences);
        }

        tokenized_texts
    }
    /// Split texts into sentences and tokens.
    ///
    /// # Arguments
    ///
    /// * `texts` - Vector of `&str`s with texts to split.
    ///
    /// Returns a vector with the same length as `texts`.
    /// Each element is a vector of sentences.
    /// Each sentence is a vector of `Token`s.
    /// Each token is a struct with fields `text` and `whitespace`.
    pub fn split(&self, texts: Vec<&str>) -> Vec<Vec<Vec<Token>>> {
        let text_lengths = texts
            .iter()
            .map(|x| x.graphemes(true).count())
            .collect::<Vec<_>>();

        let (preds, all_idx, n_cuts_per_text) = self.get_raw_preds(&texts, &text_lengths);
        let all_avg_preds = self.average_preds(preds, all_idx, n_cuts_per_text, &text_lengths);

        self.split_texts_from_preds(texts, all_avg_preds)
    }
}
