#![feature(proc_macro_hygiene)]
use inline_python::python;
use std::process::Command;
use std::sync::Once;

static INSTALL: Once = Once::new();

fn install_module() {
    INSTALL.call_once(|| {
        assert!(Command::new("maturin")
            .arg("develop")
            .status()
            .expect("failed to execute `maturin`.")
            .success())
    });
}

#[test]
fn test_splitter_model_works() {
    install_module();

    python! {
        import nnsplit

        model = nnsplit.NNSplit.load("de")
        splits = model.split(["Das ist ein Test Das ist noch ein Test."])[0]

        assert [str(x) for x in splits] == ["Das ist ein Test ", "Das ist noch ein Test."]
    }
}

#[test]
fn test_splitter_model_works_with_args() {
    install_module();

    python! {
        import nnsplit

        model = nnsplit.NNSplit.load("de", threshold=1.0)
        splits = model.split(["Das ist ein Test Das ist noch ein Test."])[0]

        assert [str(x) for x in splits] == ["Das ist ein Test Das ist noch ein Test."]
    }
}
