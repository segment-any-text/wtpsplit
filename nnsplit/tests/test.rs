// use nnsplit::{NNSplit, Token};

// macro_rules! token {
//     ($text:expr, $whitespace:expr) => (Token {
//         text: String::from($text),
//         whitespace: String::from($whitespace)
//     });
// }

// #[test]
// fn it_splits_english_correctly() -> failure::Fallible<()> {
//     let splitter = NNSplit::new("en")?;

//     let result = splitter.split(vec!["This is a test This is another test."]);

//     assert_eq!(vec![vec![
//                 vec![
//                     token!("This", " "),
//                     token!("is", " "),
//                     token!("a", " "),
//                     token!("test", " "),
//                 ],
//                 vec![
//                     token!("This", " "),
//                     token!("is", " "),
//                     token!("another", " "),
//                     token!("test", ""),
//                     token!(".", ""),
//                 ],
//             ]
//         ], result
//     );

//     Ok(())
// }

// #[test]
// fn it_splits_german_correctly() -> failure::Fallible<()> {
//     let splitter = NNSplit::new("de")?;

//     let result = splitter.split(vec!["Das ist ein Test Das ist noch ein Test."]);

//     assert_eq!(vec![vec![
//                 vec![
//                     token!("Das", " "),
//                     token!("ist", " "),
//                     token!("ein", " "),
//                     token!("Test", " "),
//                 ],
//                 vec![
//                     token!("Das", " "),
//                     token!("ist", " "),
//                     token!("noch", " "),
//                     token!("ein", " "),
//                     token!("Test", ""),
//                     token!(".", ""),
//                 ],
//             ]
//         ], result
//     );

//     Ok(())
// }

// #[test]
// fn it_splits_long_text_correctly() -> failure::Fallible<()> {
//     let splitter = NNSplit::new("en")?;

//     let result = splitter.split(vec!["Fast, robust sentence splitting with Javascript, Rust and Python bindings Punctuation is not necessary to split sentences correctly sometimes even incorrect case is split correctly."]);

//     assert_eq!(vec![vec![
//                 vec![
//                     token!("Fast", ""),
//                     token!(",", " "),
//                     token!("robust", " "),
//                     token!("sentence", " "),
//                     token!("splitting", " "),
//                     token!("with", " "),
//                     token!("Javascript", ""),
//                     token!(",", " "),
//                     token!("Rust", " "),
//                     token!("and", " "),
//                     token!("Python", " "),
//                     token!("bindings", " "),
//                 ],
//                 vec![
//                     token!("Punctuation", " "),
//                     token!("is", " "),
//                     token!("not", " "),
//                     token!("necessary", " "),
//                     token!("to", " "),
//                     token!("split", " "),
//                     token!("sentences", " "),
//                     token!("correctly", " "),
//                 ],
//                 vec![
//                     token!("sometimes", " "),
//                     token!("even", " "),
//                     token!("incorrect", " "),
//                     token!("case", " "),
//                     token!("is", " "),
//                     token!("split", " "),
//                     token!("correctly", ""),
//                     token!(".", ""),
//                 ],
//             ]
//         ], result
//     );

//     Ok(())
// }

// #[test]
// fn test_it_batches_correctly() -> failure::Fallible<()> {
//     let mut splitter = NNSplit::new("de")?;
//     splitter.with_batch_size(2);

//     let result = splitter.split(vec!["First", "Second", "Third"]);

//     assert_eq!(
//         vec![
//             vec![
//                 vec![
//                     token!("First", ""),
//                 ]
//             ],
//             vec![
//                 vec![
//                     token!("Second", ""),
//                 ]
//             ],
//             vec![
//                 vec![
//                     token!("Third", ""),
//                 ]
//             ],
//         ], result
//     );

//     Ok(())
// }
