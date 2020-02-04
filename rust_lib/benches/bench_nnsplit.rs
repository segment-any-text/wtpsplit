use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) -> failure::Fallible<()> {
    let splitter = nnsplit::NNSplit::new("de")?;

    c.bench_function("split 1", |b| b.iter(|| splitter.split(black_box(vec!["Das ist ein Test."]))));

    Ok(())
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);