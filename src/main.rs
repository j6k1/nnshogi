extern crate rand;

extern crate usiagent;
extern crate simplenn;

pub mod player;
pub mod error;
pub mod nn;

use std::error::Error;

use usiagent::UsiAgent;
use usiagent::output::USIStdErrorWriter;

use player::NNShogiPlayer;

fn main() {
	let agent = UsiAgent::new(NNShogiPlayer::new());

	match agent.start_default() {
		Ok(()) => (),
		Err(ref e) =>  {
			USIStdErrorWriter::write(e.description()).is_err();
		}
	};
}
