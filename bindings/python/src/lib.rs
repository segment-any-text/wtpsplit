#![feature(proc_macro_hygiene)]
use inline_python::python;
use ndarray::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug() {
        let array = Array::range(0., 10., 1.);
        println!("{:#?}", array.as_ptr());
        println!("{:#?}", array.len());
    }
}
