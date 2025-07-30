use picoc::opto;

fn main() {
    println!("picocuda")
}

#[derive(Error, Debug)] pub enum CompileError {
    #[error("i/o error")] IOError(#[from] io::Error),
    #[error("type error")] TypeError(#[from] TypeError)
}
pub fn compile_cfg(src: &Path) -> Result<(), CompileError> {
    let (src_c0, dst_r5) = (File::open("hello.c")?, File::create("foo.txt")?);
    let ast = parser::parse(src_c0);
    let _ = typer::typ()?;

    let opto = true;

    if opto {
        let linear = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/bril/benchmarks/core/fact.bril");
        let cfg = opto::parser::parse(&linear)?;
        // local, regional, global, program opts...
    }


    let aasmtree = selector::select(ast, CPU::R5, CallingConvention::SystemV);
    let asmtree = allocator::allocate(aasmtree);
    let machcode = encoder::encode(asmtree);
    let elf = exporter::export(machcode, Format::Executable, dst_r5);
    // TODO: write elf to disk
    Ok(())
}

fn compile_son() -> () {}

#[derive(Error, Debug)] pub enum CompileError {
    #[error("i/o error")] IOError(#[from] io::Error),
    #[error("type error")] TypeError(#[from] sema::typer::TypeError),
    #[error("parse error")] ParseError(#[from] opto::parser::ParseError)
}