use nnsplit::{NNSplit, Token};

macro_rules! token {
    ($text:expr, $whitespace:expr) => (Token {
        text: String::from($text),
        whitespace: String::from($whitespace) 
    });
}

#[test]
fn it_splits_german_correctly() -> failure::Fallible<()> {
    let splitter = NNSplit::new("../data/de/ts_cpu.pt")?;
    let result = splitter.split(vec!["Das ist ein Test Das ist noch ein Test."]);

    assert_eq!(vec![vec![
                vec![
                    token!("Das", " "), 
                    token!("ist", " "),
                    token!("ein", " "),
                    token!("Test", " "),
                ],
                vec![
                    token!("Das", " "), 
                    token!("ist", " "),
                    token!("noch", " "),
                    token!("ein", " "),
                    token!("Test", ""),
                    token!(".", ""),
                ],
            ]
        ], result
    );

    Ok(())
}