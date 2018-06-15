extern crate rand;
extern crate getopts;

extern crate usiagent;
extern crate simplenn;

pub mod player;
pub mod error;
pub mod nn;
pub mod hash;

use std::error::Error;
use std::env;
use std::sync::Mutex;
use std::sync::Arc;
use std::io::{ BufWriter, Write  };
use std::fs;
use std::fs::OpenOptions;

use getopts::Options;

use usiagent::UsiAgent;
use usiagent::selfmatch::*;
use usiagent::output::*;
use usiagent::event::*;
use usiagent::command::*;
use usiagent::shogi::*;
use usiagent::error::*;
use usiagent::player::*;
use usiagent::TryToString;

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
	let args: Vec<String> = env::args().collect();
	let mut opts = Options::new();
	opts.optflag("l", "learn", "Self-game mode.");

	let matches = match opts.parse(&args[1..]) {
		Ok(m) => m,
		Err(ref e) => {
			return Err(ApplicationError::StartupError(e.to_string()));
		}
	};

	if matches.opt_present("l") {
		let info_sender_arc = Arc::new(Mutex::new(CosoleInfoSender::new()));

		let mut engine = SelfMatchEngine::new(
			NNShogiPlayer::new(String::from("nn.a.bin"),String::from("nn.b.bin")),
			NNShogiPlayer::new(String::from("nn_opponent.a.bin"),String::from("nn_opponent.b.bin")),
			info_sender_arc,
			UsiGoTimeLimit::Limit(Some((10 * 1000, 10 * 1000)),None),
			None,Some(10)
		);

		let mut flip = false;

		let on_before_newgame = move || {
			flip = !flip;
			!flip
		};

		let system_event_queue = engine.system_event_queue.clone();

		let input_read_handler = move |input| {
			if input == "quit" {
				return match system_event_queue.lock()  {
					Ok(mut system_event_queue) => {
						system_event_queue.push(SystemEvent::Quit);
						Ok(())
					},
					Err(_) => {
						Err(SelfMatchRunningError::InvalidState(String::from(
							"Failed to secure exclusive lock of system_event_queue."
						)))
					}
				};
			}
			Ok(())
		};

		let r = engine.start_default(|self_match_event_dispatcher| {
									self_match_event_dispatcher
										.add_handler(SelfMatchEventKind::GameStart,
														Box::new(move |_,e| {
											match e {
												&SelfMatchEvent::GameStart(n,_) => {
													print!("プレイヤー{}が先手で開始しました。\n",n);
													Ok(())
												},
												e => Err(EventHandlerError::InvalidState(e.event_kind())),
											}
										}));
									self_match_event_dispatcher
										.add_handler(SelfMatchEventKind::Moved,
														Box::new(move |_,e| {
											match e {
												&SelfMatchEvent::Moved(t,m) => {
													print!("Move: {:?}n",m);
													Ok(())
												},
												e => Err(EventHandlerError::InvalidState(e.event_kind())),
											}
										}));
									self_match_event_dispatcher
										.add_handler(SelfMatchEventKind::GameEnd,
														Box::new(move |_,e| {
											print!("ゲームが終了しました。{:?}\n",e);
											Ok(())
										}));
									self_match_event_dispatcher
										.add_handler(SelfMatchEventKind::Abort,
														Box::new(move |_,e| {
											match e {
												&SelfMatchEvent::Abort => {
													print!("思考が途中で中断されました。\n");
													Ok(())
												},
												e => Err(EventHandlerError::InvalidState(e.event_kind())),
											}
										}));
								},
								on_before_newgame,
								None,Some(KifuWriter::new(String::from("logs/kifu.txt"))?),
								input_read_handler,
								[
									("BaseDepth",SysEventOption::Num(1)),
									("MaxDepth",SysEventOption::Num(3)),
								].into_iter().map(|&(ref k,ref v)| {
									(k.to_string(),v.clone())
								}).collect::<Vec<(String,SysEventOption)>>(),
								[
									("BaseDepth",SysEventOption::Num(1)),
									("MaxDepth",SysEventOption::Num(3)),
								].into_iter().map(|&(ref k,ref v)| {
									(k.to_string(),v.clone())
								}).collect::<Vec<(String,SysEventOption)>>(),
								|on_error_handler,e| {
									match on_error_handler {
										Some(ref h) => {
											h.lock().map(|h| h.call(e)).is_err();
										},
										None => (),
									}
								});

		r.map_err(|_| ApplicationError::SelfMatchRunningError(String::from(
			"自己対局の実行中にエラーが発生しました。詳細はログを参照してください..."
		)))
	} else {
		let agent = UsiAgent::new(NNShogiPlayer::new(String::from("nn.a.bin"),String::from("nn.b.bin")));

		let r = agent.start_default(|on_error_handler,e| {
			match on_error_handler {
				Some(ref h) => {
					h.lock().map(|h| h.call(e)).is_err();
				},
				None => (),
			}
		});
		r.map_err(|_| ApplicationError::AgentRunningError(String::from(
			"USIAgentの実行中にエラーが発生しました。詳細はログを参照してください..."
		)))
	}
}
struct CosoleInfoSender {

}
impl CosoleInfoSender {
	pub fn new() -> CosoleInfoSender {
		CosoleInfoSender {}
	}
}
impl InfoSender for CosoleInfoSender {
	fn send(&mut self,commands:Vec<UsiInfoSubCommand>) -> Result<(), InfoSendError> {
		for command in commands {
			print!("{}\n",command.try_to_string()?);
		}
		Ok(())
	}
}
#[derive(Debug)]
pub struct KifuWriter {
	writer:BufWriter<fs::File>,
}
impl KifuWriter {
	pub fn new(file:String) -> Result<KifuWriter,ApplicationError> {
		Ok(KifuWriter {
			writer:BufWriter::new(OpenOptions::new().append(true).create(true).open(file)?),
		})
	}
}
impl SelfMatchKifuWriter<SelfMatchRunningError> for KifuWriter {
	fn write(&mut self,initial_sfen:&String,m:&Vec<Move>) -> Result<(),SelfMatchRunningError> {
		let sfen = match self.to_sfen(initial_sfen,m) {
			Err(ref e) => {
				return Err(SelfMatchRunningError::InvalidState(e.to_string()));
			},
			Ok(sfen) => sfen,
		};

		match self.writer.write(sfen.as_bytes()) {
			Err(ref e) => {
				Err(SelfMatchRunningError::InvalidState(e.to_string()))
			},
			Ok(_) => Ok(()),
		}
	}
}