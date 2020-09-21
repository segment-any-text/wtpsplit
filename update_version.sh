if [ `uname` = "Darwin" ]
then
    SED=gsed
else
    SED=sed
fi

function update_cargo_toml_version {
    VERSION=$1
    FILE=$2

    $SED -i "0,/^version/s/^version *= *\".*\"/version = \"$VERSION\"/" $FILE
}

update_cargo_toml_version $1 nnsplit/Cargo.toml
update_cargo_toml_version $1 bindings/python/Cargo.toml
update_cargo_toml_version $1-post0 bindings/python/Cargo.build.toml
npm version $1 --prefix bindings/javascript --allow-same-version
