extern crate rand;
extern crate getopts;
extern crate toml;
#[macro_use]
extern crate serde_derive;

extern crate usiagent;
extern crate simplenn;

pub mod player;
pub mod error;
pub mod nn;
pub mod hash;

use std::env;
use std::sync::Mutex;
use std::sync::Arc;
use std::io::{ BufWriter, Write, BufReader, Read };
use std::fs;
use std::fs::OpenOptions;
use std::fs::File;
use std::path::Path;
use std::time::Duration;

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

#[derive(Debug, Deserialize)]
pub struct Config {
	base_depth:Option<u32>,
	max_depth:Option<u32>,
	time_limit:Option<u32>,
	running_time:Option<u32>,
	number_of_games:Option<u32>,
}
pub struct ConfigLoader {
	reader:BufReader<File>,
}
impl ConfigLoader {
	pub fn new (file:&str) -> Result<ConfigLoader, ApplicationError> {
		match Path::new(file).exists() {
			true => {
				Ok(ConfigLoader {
					reader:BufReader::new(OpenOptions::new().read(true).create(false).open(file)?),
				})
			},
			false => {
				Err(ApplicationError::StartupError(String::from(
					"設定ファイルが見つかりません。"
				)))
			}
		}
	}
	pub fn load(&mut self) -> Result<Config,ApplicationError> {
		let mut buf = String::new();
		self.reader.read_to_string(&mut buf)?;
		match toml::from_str(buf.as_str()) {
			Ok(r) => Ok(r),
			Err(ref e) => {
				USIStdErrorWriter::write(&e.to_string()).is_err();
				Err(ApplicationError::StartupError(String::from(
					"設定ファイルのロード時にエラーが発生しました。"
				)))
			}
		}
	}
}
fn main() {
	match run() {
		Ok(()) => (),
		Err(ref e) =>  {
			USIStdErrorWriter::write(&e.to_string()).is_err();
		}
	};
}
fn run() -> Result<(),ApplicationError> {
	let args: Vec<String> = env::args().collect();
	let mut opts = Options::new();
	opts.optflag("l", "learn", "Self-game mode.");
	opts.optopt("", "basedepth", "Search-default-depth.", "number of depth");
	opts.optopt("", "maxdepth", "Search-max-depth.", "number of depth");
	opts.optopt("", "timelimit", "USI time limit.", "milli second");
	opts.optopt("t", "time", "Running time.", "second");
	opts.optopt("c", "count", "execute game count.", "number of game count");

	let matches = match opts.parse(&args[1..]) {
		Ok(m) => m,
		Err(ref e) => {
			return Err(ApplicationError::StartupError(e.to_string()));
		}
	};

	if matches.opt_present("l") {
		let config = ConfigLoader::new("settings.toml")?.load()?;

		let base_depth = match config.base_depth {
			Some(base_depth) if base_depth > 0 => base_depth,
			_ => {
				return Err(ApplicationError::StartupError(String::from(
					"base_depthの設定値が不正です。"
				)))
			}
		};

		let base_depth:u32 = match matches.opt_str("basedepth") {
			Some(base_depth) => base_depth.parse()?,
			None => base_depth,
		};

		let max_depth = match config.max_depth {
			Some(max_depth) if max_depth > 0 => max_depth,
			_ => {
				return Err(ApplicationError::StartupError(String::from(
					"base_depthの設定値が不正です。"
				)))
			}
		};

		let max_depth:u32 = match matches.opt_str("maxdepth") {
			Some(max_depth) => max_depth.parse()?,
			None => max_depth,
		};

		let time_limit = config.time_limit.map_or(UsiGoTimeLimit::Infinite, |l| {
				if l == 0 {
					UsiGoTimeLimit::Infinite
				} else {
					UsiGoTimeLimit::Limit(Some((l,l)),None)
				}
			});

		let time_limit:UsiGoTimeLimit = match matches.opt_str("timelimit") {
			Some(time_limit) => {
				let l = time_limit.parse()?;
				UsiGoTimeLimit::Limit(Some((l,l)),None)
			},
			None => time_limit,
		};

		let running_time = config.running_time.map_or(None,|t| {
				if t == 0 {
					None
				} else {
					Some(Duration::from_millis(t as u64 * 1000))
				}
			});

		let running_time:Option<Duration> = match matches.opt_str("t") {
			Some(running_time) => Some(Duration::from_millis(running_time.parse::<u64>()? * 1000)),
			None => running_time,
		};

		let number_of_games = config.number_of_games.map_or(None,|t| {
				if t == 0 {
					None
				} else {
					Some(t)
				}
			});

		let number_of_games:Option<u32> = match matches.opt_str("c") {
			Some(number_of_games) => Some(number_of_games.parse()?),
			None => number_of_games,
		};

		print!("base_depth = {:?}, max_depth = {:?}, time_limit = {:?}, running_time = {:?}, number_of_games = {:?}",
			base_depth, max_depth, time_limit, running_time, number_of_games
		);

		let info_sender_arc = Arc::new(Mutex::new(CosoleInfoSender::new()));

		let mut engine = SelfMatchEngine::new(
			NNShogiPlayer::new(String::from("nn.a.bin"),String::from("nn.b.bin")),
			NNShogiPlayer::new(String::from("nn_opponent.a.bin"),String::from("nn_opponent.b.bin")),
			info_sender_arc,
			time_limit,
			running_time,number_of_games
		);

		let mut flip = true;

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
													match t {
														Teban::Sente => {
															print!("先手: {}\n",m);
														},
														Teban::Gote => {
															print!("後手: {}\n",m);
														}
													}
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
									("BaseDepth",SysEventOption::Num(base_depth)),
									("MaxDepth",SysEventOption::Num(max_depth)),
								].into_iter().map(|&(ref k,ref v)| {
									(k.to_string(),v.clone())
								}).collect::<Vec<(String,SysEventOption)>>(),
								[
									("BaseDepth",SysEventOption::Num(base_depth)),
									("MaxDepth",SysEventOption::Num(max_depth)),
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
		std::io::stdout().flush().is_err();
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

		match self.writer.write(format!("{}\n",sfen).as_bytes()) {
			Err(ref e) => {
				Err(SelfMatchRunningError::InvalidState(e.to_string()))
			},
			Ok(_) => Ok(()),
		}
	}
}