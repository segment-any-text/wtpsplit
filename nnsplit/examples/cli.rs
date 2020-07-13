use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let lang = env::args().nth(1).unwrap();
    let text = env::args().nth(2).unwrap();

    let splitter =
        nnsplit::NNSplit::load(&lang, tch::Device::Cpu, nnsplit::NNSplitOptions::default())?;

    let input: Vec<&str> = vec![&text]; // input can consist of multiple texts to allow parallelization
    let splits = &splitter.split(&input)[0];

    for sentence in splits.iter() {
        println!("{}", sentence.text());
    }

    Ok(())
}
