extern crate rand;

extern crate usiagent;
extern crate simplenn;

pub mod player;
pub mod error;
pub mod nn;
pub mod hash;

use std::error::Error;

use usiagent::UsiAgent;
use usiagent::output::USIStdErrorWriter;

use player::NNShogiPlayer;
use error::ApplicationError;

fn main() {
	match run() {
		Ok(()) => (),
		Err(ref e) =>  {
			USIStdErrorWriter::write(e.description()).is_err();
		}
	};
}
fn run() -> Result<(),ApplicationError> {
	let agent = UsiAgent::new(NNShogiPlayer::new(String::from("nn.a.bin"),String::from("nn.b.bin")));

	let r = agent.start_default(|on_error_handler,e| {
		match on_error_handler {
			Some(ref h) => {
				h.lock().map(|h| h.call(e)).is_err();
			},
			None => (),
		}
	});
	r.map_err(|_| ApplicationError::StartupError(String::from(
		"Startup failed."
	)))
}