use std::path::Path;
use std::vec::Vec;
use std::ops::Range;
use std::cmp;
use std::convert::{TryInto, TryFrom};

use ndarray::prelude::*;
use ndarray::Array2;
use tch::{Tensor, Device};
use unicode_segmentation::UnicodeSegmentation;

fn text_to_id(character: &str) -> u8 {
    if character.chars().count() != 1 {
        return 1
    }
    
    let ord = character.chars().next().unwrap() as u32;
    if ord <= 127 { u8::try_from(ord + 2).unwrap() } else { 1 }
}

#[derive(Debug, PartialEq)]
pub struct Token {
    pub text: String,
    pub whitespace: String
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
            whitespace: text[last_char_index..].to_string()
        }
    }
}

pub struct NNSplit {
    model: tch::CModule,
    threshold: f32,
    stride: usize,
    cut_length: usize,
    batch_size: usize,
    device: Device,
}

impl NNSplit {
    const THRESHOLD: f32 = 0.5;
    const STRIDE: usize = 50;
    const CUT_LENGTH: usize = 100;
    const BATCH_SIZE: usize = 32;

    pub fn new(model_name: &str) -> failure::Fallible<NNSplit> {
        let model = tch::CModule::load(model_name)?;
        let device = Device::cuda_if_available();

        Ok(NNSplit { 
            model, 
            threshold: NNSplit::THRESHOLD, 
            stride: NNSplit::STRIDE, 
            cut_length: NNSplit::CUT_LENGTH,
            batch_size: NNSplit::BATCH_SIZE,
            device
        })
    }
    
    pub fn from_model(model: tch::CModule) -> failure::Fallible<NNSplit> {
        let device = Device::cuda_if_available();

        Ok(NNSplit { 
            model, 
            threshold: NNSplit::THRESHOLD, 
            stride: NNSplit::STRIDE, 
            cut_length: NNSplit::CUT_LENGTH, 
            batch_size: NNSplit::BATCH_SIZE,
            device
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

    pub fn split(&self, texts: Vec<&str>) -> Vec<Vec<Vec<Token>>> {
        let mut all_inputs: Vec<u8> = Vec::new();
        let mut all_idx: Vec<Range<usize>> = Vec::new();
        let mut n_cuts_per_text: Vec<usize> = Vec::new();
        let text_lengths = texts.iter().map(|x| x.graphemes(true).count()).collect::<Vec<_>>();
        let mut total_cuts: usize = 0;

        for (text, length) in texts.iter().zip(text_lengths.iter()) {
            let current_length = cmp::max(*length, self.cut_length);
            let mut inputs = vec![0; current_length];
            
            for (i, character) in text.graphemes(true).enumerate() {
                inputs[i] = text_to_id(character);
            }

            let mut start = 0;
            let mut end = 0;
            let mut i: usize = 0;

            while end != current_length {
                end = cmp::min(start + self.cut_length, current_length);
                start = end - self.cut_length;

                let range = Range { start, end };
                all_inputs.extend(inputs[range.clone()].iter()); // TODO: maybe better with slices?
                all_idx.push(range);
                
                start += self.stride;
                i += 1;
            }
            
            total_cuts += i;
            n_cuts_per_text.push(i);
        }

        let batched_inputs = Tensor::of_slice(&all_inputs).view((-1, self.cut_length as i64));
        let preds = self.model.forward_ts(&[batched_inputs]).unwrap().sigmoid(); // TODO: batch properly
        let preds : ArrayD<f32> = (&preds).try_into().unwrap();

        let mut all_avg_preds = text_lengths.iter().map(|x| Array2::<f32>::zeros((*x, 2))).collect::<Vec<_>>();
        let mut all_avg_pred_counts = text_lengths.iter().map(|x| Array2::<f32>::zeros((*x, 1))).collect::<Vec<_>>();

        let mut current_text = 0;
        let mut current_i = 0;

        // TODO: maybe less slices?
        for i in 0..total_cuts {
            let current_preds = &mut all_avg_preds[current_text];
            let current_pred_counts = &mut all_avg_pred_counts[current_text];

            let range = Range { 
                start: all_idx[i].start, 
                end: cmp::min(all_idx[i].end, text_lengths[current_text]) 
            };

            let mut slice = current_preds.slice_mut(s![range.clone(), ..]);
            slice += &preds.slice(s![i, 0..text_lengths[current_text], ..]);

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

        let mut tokenized_texts: Vec<Vec<Vec<Token>>> = Vec::new();
        for (avg_preds, text) in all_avg_preds.iter().zip(texts.iter()) {
            let mut sentences: Vec<Vec<Token>> = Vec::new();
            let mut tokens: Vec<Token> = Vec::new();
            let mut token = String::new();

            for (i, character) in text.graphemes(true).enumerate() {
                token += character;

                if avg_preds[[i, 0]] > self.threshold {
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
}
