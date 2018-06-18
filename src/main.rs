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

use std::env;
use std::sync::Mutex;
use std::sync::Arc;
use std::io::{ Write, BufReader, Read };
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
	running_time:Option<String>,
	number_of_games:Option<u32>,
	silent:bool,
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
	opts.optopt("t", "time", "Running time.", "s: second, m: minute, h: hour, d: day");
	opts.optopt("c", "count", "execute game count.", "number of game count");
	opts.optflag("", "silent", "silent mode.");

	let matches = match opts.parse(&args[1..]) {
		Ok(m) => m,
		Err(ref e) => {
			return Err(ApplicationError::StartupError(e.to_string()));
		}
	};

	if matches.opt_present("l") {
		let config = ConfigLoader::new("settings.toml")?.load()?;

		let silent =  matches.opt_present("silent") || config.silent;

		let base_depth = match config.base_depth {
			Some(base_depth) => base_depth,
			_ => {
				return Err(ApplicationError::StartupError(String::from(
					"base_depthの値が未設定です。"
				)))
			}
		};

		let base_depth:u32 = match matches.opt_str("basedepth") {
			Some(base_depth) => base_depth.parse()?,
			None => base_depth,
		};

		if base_depth <= 0 {
			return Err(ApplicationError::StartupError(String::from(
				"base_depthの設定値が不正です。"
			)));
		}

		let max_depth = match config.max_depth {
			Some(max_depth) => max_depth,
			_ => {
				return Err(ApplicationError::StartupError(String::from(
					"base_depthが未設定です。"
				)));
			}
		};

		let max_depth:u32 = match matches.opt_str("maxdepth") {
			Some(max_depth) => max_depth.parse()?,
			None => max_depth,
		};

		if max_depth <= 0 {
			return Err(ApplicationError::StartupError(String::from(
				"max_depthの設定値が不正です。"
			)));
		}

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
				if l == 0 {
					UsiGoTimeLimit::Infinite
				} else {
					UsiGoTimeLimit::Limit(Some((l,l)),None)
				}
			}
			None => time_limit,
		};

		let running_time = config.running_time.map_or(None,|t| {
			if t == "" || t == "0" || t == "0s" || t == "0m" || t == "0h" || t == "0d" {
				None
			} else {
				Some(t)
			}
		});

		let running_time:Option<String> = match matches.opt_str("t") {
			Some(t) => {
				if t == "" || t == "0" || t == "0s" || t == "0m" || t == "0h" || t == "0d" {
					None
				} else {
					Some(t)
				}
			},
			None => running_time,
		};

		let running_time_none_parsed = match running_time {
			None => String::from("0s"),
			Some(ref running_time) => running_time.clone(),
		};

		let running_time = match running_time {
			None => None,
			Some(ref running_time) if running_time.ends_with("s") => {
				let len = running_time.chars().count();
				let s = running_time.chars().take(len-1).collect::<String>();
				Some(Duration::from_secs(s.parse::<u64>()?))
			},
			Some(ref running_time) if running_time.ends_with("m") => {
				let len = running_time.chars().count();
				let s = running_time.chars().take(len-1).collect::<String>();
				Some(Duration::from_secs(60 * s.parse::<u64>()?))
			},
			Some(ref running_time) if running_time.ends_with("h") => {
				let len = running_time.chars().count();
				let s = running_time.chars().take(len-1).collect::<String>();
				Some(Duration::from_secs(60 * 60 * s.parse::<u64>()?))
			},
			Some(ref running_time) if running_time.ends_with("d") => {
				let len = running_time.chars().count();
				let s = running_time.chars().take(len-1).collect::<String>();
				Some(Duration::from_secs(24 * 60 * 60 * s.parse::<u64>()?))
			},
			Some(ref running_time) => {
				Some(Duration::from_secs(60 * running_time.parse::<u64>()?))
			}
		};

		let number_of_games = config.number_of_games.map_or(None,|t| {
			if t == 0 {
				None
			} else {
				Some(t)
			}
		});

		let number_of_games:Option<u32> = match matches.opt_str("c") {
			Some(number_of_games) => {
				let n = number_of_games.parse()?;
				if n == 0 {
					None
				} else {
					Some(n)
				}
			},
			None => number_of_games,
		};

		print!("base_depth = {:?}, max_depth = {:?}, time_limit = {:?}, running_time = {:?}, number_of_games = {:?}",
			base_depth, max_depth, time_limit, running_time_none_parsed, number_of_games
		);

		let info_sender_arc = Arc::new(Mutex::new(CosoleInfoSender::new(silent)));

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
										.add_handler(SelfMatchEventKind::GameStart, move |_,e| {
											match e {
												&SelfMatchEvent::GameStart(n,t,_) => {
													let t = match t {
														Teban::Sente => String::from("先手"),
														Teban::Gote => String::from("後手"),
													};
													print!("プレイヤー{}が{}で開始しました。\n",n,t);
													Ok(())
												},
												e => Err(EventHandlerError::InvalidState(e.event_kind())),
											}
										});
									self_match_event_dispatcher
										.add_handler(SelfMatchEventKind::Moved, move |_,e| {
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
										});
									self_match_event_dispatcher
										.add_handler(SelfMatchEventKind::GameEnd, move |_,e| {
											match *e {
												SelfMatchEvent::GameEnd(SelfMatchGameEndState::Win(t)) => {
													let t = match t {
														Teban::Sente => String::from("先手"),
														Teban::Gote => String::from("後手"),
													};
													print!("{}の勝ちです。\n",t);
												},
												SelfMatchEvent::GameEnd(SelfMatchGameEndState::Draw) => {
													print!("引き分けです。\n");
												},
												SelfMatchEvent::GameEnd(SelfMatchGameEndState::Resign(t)) => {
													let t = match t {
														Teban::Sente => String::from("先手"),
														Teban::Gote => String::from("後手"),
													};
													print!("{}の投了です。\n",t);
												},
												SelfMatchEvent::GameEnd(SelfMatchGameEndState::NyuGyokuWin(t)) => {
													let t = match t {
														Teban::Sente => String::from("先手"),
														Teban::Gote => String::from("後手"),
													};
													print!("{}の入玉宣言勝ちです。\n",t);
												},
												SelfMatchEvent::GameEnd(SelfMatchGameEndState::NyuGyokuLose(t)) => {
													let t = match t {
														Teban::Sente => String::from("先手"),
														Teban::Gote => String::from("後手"),
													};
													print!("{}が入玉宣言しましたが、成立しませんでした（{}の負けです）。\n",t,t);
												},
												SelfMatchEvent::GameEnd(SelfMatchGameEndState::Foul(t,FoulKind::InvalidMove)) => {
													let t = match t {
														Teban::Sente => String::from("先手"),
														Teban::Gote => String::from("後手"),
													};
													print!("{}の反則負けです（不正な手）\n",t);
												},
												SelfMatchEvent::GameEnd(SelfMatchGameEndState::Foul(t,FoulKind::PutFuAndMate)) => {
													let t = match t {
														Teban::Sente => String::from("先手"),
														Teban::Gote => String::from("後手"),
													};
													print!("{}の反則負けです（打ち歩詰め））\n",t);
												},
												SelfMatchEvent::GameEnd(SelfMatchGameEndState::Foul(t,FoulKind::Sennichite)) => {
													let t = match t {
														Teban::Sente => String::from("先手"),
														Teban::Gote => String::from("後手"),
													};
													print!("{}の反則負けです（千日手）\n",t);
												},
												SelfMatchEvent::GameEnd(SelfMatchGameEndState::Foul(t,FoulKind::SennichiteOu)) => {
													let t = match t {
														Teban::Sente => String::from("先手"),
														Teban::Gote => String::from("後手"),
													};
													print!("{}の反則負けです（連続王手の千日手）\n",t);
												},
												SelfMatchEvent::GameEnd(SelfMatchGameEndState::Foul(t,FoulKind::NotRespondedOute)) => {
													let t = match t {
														Teban::Sente => String::from("先手"),
														Teban::Gote => String::from("後手"),
													};
													print!("{}の反則負けです（王手に応じなかった）\n",t);
												},
												SelfMatchEvent::GameEnd(SelfMatchGameEndState::Timeover(t)) => {
													let t = match t {
														Teban::Sente => String::from("先手"),
														Teban::Gote => String::from("後手"),
													};
													print!("{}が制限時間を超過しました。（負け）\n",t);
												},
												ref e => {
													return Err(EventHandlerError::InvalidState(e.event_kind()));
												}
											}
											Ok(())
										});
									self_match_event_dispatcher
										.add_handler(SelfMatchEventKind::Abort, move |_,e| {
											match e {
												&SelfMatchEvent::Abort => {
													print!("思考が途中で中断されました。\n");
													Ok(())
												},
												e => Err(EventHandlerError::InvalidState(e.event_kind())),
											}
										});
								},
								on_before_newgame,
								None,Some(FileSfenKifuWriter::new(String::from("logs/kifu.txt"))?),
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
		r.map_err(|_| ApplicationError::SelfMatchRunningError(
						SelfMatchRunningError::InvalidState(String::from(
			"自己対局の実行中にエラーが発生しました。詳細はログを参照してください..."
		))))
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
	silent:bool,
}
impl CosoleInfoSender {
	pub fn new(silent:bool) -> CosoleInfoSender {
		CosoleInfoSender {
			silent:silent
		}
	}
}
impl InfoSender for CosoleInfoSender {
	fn send(&mut self,commands:Vec<UsiInfoSubCommand>) -> Result<(), InfoSendError> {
		if !self.silent {
			for command in commands {
				print!("{}\n",command.try_to_string()?);
			}
		}
		Ok(())
	}
}