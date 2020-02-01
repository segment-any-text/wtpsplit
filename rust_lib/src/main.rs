extern crate tch;

use std::path::Path;
use std::vec::Vec;
use std::ops::Range;
use std::cmp;
use std::convert::{TryInto,TryFrom};

fn text_to_id(character: char) -> u8 {
    let ord = character as u32;
    if ord <= 127 { u8::try_from(ord + 2).unwrap() } else { 1 }
}

struct NNSplit {
    model: tch::CModule,
    threshold: f32,
    stride: usize,
    cut_length: usize,
    // TODO: add device
}

impl NNSplit {
    pub fn new(model_name_or_path: &str) -> failure::Fallible<NNSplit> {
        let model;

        if Path::new(model_name_or_path).exists() || model_name_or_path.contains(".") {
            model = tch::CModule::load(model_name_or_path)?;
        } else {
            // TODO: implement this
            model = tch::CModule::load(model_name_or_path)?;
        }

        Ok(NNSplit { model, threshold: 0.5, stride: 50, cut_length: 100 })
    }

    pub fn split(&self, texts: Vec<&str>) {
        let max_length = 4000usize;
        let batch_size = 32;

        let mut all_inputs: Vec<u8> = Vec::new();
        let mut all_idx: Vec<Range<usize>> = Vec::new();
        let mut n_cuts_per_text: Vec<i32> = Vec::new();

        let raw_max_length = texts.iter().map(|x| x.chars().count()).max().unwrap(); 
        let max_effective_length = cmp::min(max_length, raw_max_length);
        let max_effective_length = cmp::max(max_effective_length, self.cut_length);

        for text in &texts {
            let current_length = cmp::max(text.chars().count(), max_effective_length);
            let mut inputs = vec![0; current_length];
            
            for (i, character) in text.chars().enumerate() {
                inputs[i] = text_to_id(character);
            }

            let mut start = 0;
            let mut end = 0;
            let mut i = 0;

            while end != current_length {
                end = cmp::min(start + self.cut_length, current_length);
                start = end - self.cut_length;

                let range = Range { start, end };
                all_inputs.extend(inputs[range.clone()].iter()); // TODO: maybe better with slices?
                all_idx.push(range);
                
                start += self.stride;
                i += 1;
            }
        }
    }
}

fn main() -> failure::Fallible<()> {
    let splitter = NNSplit::new("../data/de.pt")?;
    splitter.split(vec!["Das ist ein Test."]);

    Ok(())
}