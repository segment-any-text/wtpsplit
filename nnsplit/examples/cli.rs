use std::env;

fn main() -> failure::Fallible<()> {
    let text = env::args().skip(1).next().unwrap();

    let model = tch::CModule::load("../data/torchscript_cpu_model.pt")?;
    let backend = nnsplit::TchRsBackend::new(model, tch::Device::Cpu, 32);

    let splitter = nnsplit::NNSplit::new(
        Box::new(backend) as Box<dyn nnsplit::Backend>,
        nnsplit::NNSplitOptions::default(),
    )?;

    let input: Vec<&str> = vec![&text]; // input can consist of multiple texts to allow parallelization
    let splits = &splitter.split(input)[0];

    for sentence in splits.iter() {
        println!("{}", sentence.text());
    }

    Ok(())
}
