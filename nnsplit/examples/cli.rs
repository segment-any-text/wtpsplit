use std::env;

fn main() -> failure::Fallible<()> {
    let text = env::args().skip(1).next().unwrap();
    let language = env::args().skip(2).next().unwrap();

    let splitter = nnsplit::NNSplit::new(&language)?;

    let input: Vec<&str> = vec![&text]; // input can consist of multiple texts to allow parallelization
    println!("{:#?}", splitter.split(input));

    Ok(())
}
