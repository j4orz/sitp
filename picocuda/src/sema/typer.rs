use thiserror::Error;
#[derive(Error, Debug)] pub enum TypeError { #[error("use not found")] UseNotFound }
pub fn typ() -> Result<(), TypeError> {
    todo!()
}