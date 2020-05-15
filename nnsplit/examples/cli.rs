use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let text = env::args().skip(1).next().unwrap();

    let splitter = nnsplit::NNSplit::new(
        "../data/torchscript_cpu_model.pt",
        tch::Device::Cpu,
        nnsplit::NNSplitOptions::default(),
    )?;

    let input: Vec<&str> = vec![&text]; // input can consist of multiple texts to allow parallelization
    let splits = &splitter.split(&input)?[0];

    for sentence in splits.iter() {
        println!("{}", sentence.text());
    }

    Ok(())
}
