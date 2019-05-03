extern crate rand;
extern crate rand_xorshift;
extern crate statrs;
extern crate getopts;
extern crate toml;
#[macro_use]
extern crate serde_derive;

extern crate usiagent;
extern crate simplenn;
extern crate csaparser;

pub mod player;
pub mod error;
pub mod nn;
pub mod learning;

use std::env;
use std::sync::Mutex;
use std::sync::Arc;
use std::io::{ Write, BufReader, Read, BufRead };
use std::fs::OpenOptions;
use std::fs::File;
use std::path::Path;
use std::time::Duration;
use std::collections::HashMap;
use rand::Rng;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

use getopts::Options;

use usiagent::UsiAgent;
use usiagent::selfmatch::*;
use usiagent::output::*;
use usiagent::event::*;
use usiagent::command::*;
use usiagent::shogi::*;
use usiagent::rule::*;
use usiagent::protocol::*;
use usiagent::error::*;
use usiagent::player::*;

use player::NNShogiPlayer;
use error::ApplicationError;
use learning::CsaLearnener;

#[derive(Debug, Deserialize)]
pub struct Config {
	base_depth:Option<u32>,
	max_depth:Option<u32>,
	time_limit:Option<u32>,
	running_time:Option<String>,
	number_of_games:Option<u32>,
	silent:bool,
	initial_position:Option<InitialPositionKifu>,
}
#[derive(Debug, Deserialize)]
pub struct InitialPositionKifu {
	path:Option<String>,
	from_last:Option<u32>,
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
	opts.optflag("", "last", "Back a few hands from the end.");
	opts.optopt("", "fromlast", "Number of moves of from the end.", "move count.");
	opts.optopt("", "kifudir", "Directory of game data to be used of learning.", "path string.");
	opts.optopt("", "lowerrate", "Lower limit of the player rate value of learning target games.", "number of rate.");

	let matches = match opts.parse(&args[1..]) {
		Ok(m) => m,
		Err(ref e) => {
			return Err(ApplicationError::StartupError(e.to_string()));
		}
	};

	if let Some(kifudir) = matches.opt_str("kifudir") {
		let lowerrate:f64 = matches.opt_str("lowerrate").unwrap_or(String::from("3000.0")).parse()?;
		CsaLearnener::new().learning(kifudir,lowerrate)
	} else if matches.opt_present("l") {
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
				UsiGoTimeLimit::Limit(None,Some(UsiGoByoyomiOrInc::Byoyomi(l)))
			}
		});

		let time_limit:UsiGoTimeLimit = match matches.opt_str("timelimit") {
			Some(time_limit) => {
				let l = time_limit.parse()?;
				if l == 0 {
					UsiGoTimeLimit::Infinite
				} else {
					UsiGoTimeLimit::Limit(None,Some(UsiGoByoyomiOrInc::Byoyomi(l)))
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

		let initial_position = if matches.opt_present("last") {
			match config.initial_position {
				None => {
					return Err(ApplicationError::StartupError(String::from(
						"initial_positionが未設定です。"
					)));
				},
				Some(ref initial_position) => {
					let fromlast = match matches.opt_str("fromlast") {
						Some(fromlast) => Some(fromlast.parse::<u32>()?),
						None => None,
					};

					match fromlast.or(initial_position.from_last) {
						None => {
							return Err(ApplicationError::StartupError(String::from(
								"initial_position.from_last及び--fromlastオプションが未設定です。"
							)));
						},
						Some(fromlast) if fromlast > 0 => {
							match initial_position.path {
								None => {
									return Err(ApplicationError::StartupError(String::from(
										"initial_position.pathが未設定です。"
									)));
								},
								Some(ref path) => Some((fromlast, path.clone())),
							}
						},
						Some(_) => {
							return Err(ApplicationError::StartupError(String::from(
								"from_lastまたは--fromlastオプションの値には0より大きい値を指定してください。"
							)));
						}
					}
				}
			}
		} else {
			None
		};

		let initial_position_creator = match initial_position {
			None => None,
			Some((fromlast, path)) => {
				let mut reader = BufReader::new(
									OpenOptions::new()
										.read(true).create(false).open(&*path)?);

				let mut sfen_list:Vec<String> = Vec::new();

				let mut buf = String::new();

				let position_parser = PositionParser::new();

				loop {
					let n = reader.read_line(&mut buf)?;
					buf = buf.trim().to_string();

					if n == 0  {
						break;
					}

					let (mut teban, mut banmen, mut mc, _, mut mvs) = match position_parser.parse(&buf.split(" ").collect::<Vec<&str>>()) {
						Ok(mut position) => match position {
							SystemEvent::Position(teban, p, n, m) => {
								let(banmen,mc) = match p {
									UsiInitialPosition::Startpos => {
										(BANMEN_START_POS.clone(), MochigomaCollections::Pair(HashMap::new(),HashMap::new()))
									},
									UsiInitialPosition::Sfen(ref b,MochigomaCollections::Pair(ref ms,ref mg)) => {
										(b.clone(),MochigomaCollections::Pair(ms.clone(),mg.clone()))
									},
									UsiInitialPosition::Sfen(ref b,MochigomaCollections::Empty) => {
										(b.clone(),MochigomaCollections::Pair(HashMap::new(),HashMap::new()))
									}
								};
								(teban,banmen,mc,n,m)
							},
							_ => {
								return Err(ApplicationError::StartupError(String::from(
									"棋譜ファイルのパース結果が不正です。"
								)));
							}
						},
						Err(_) => {
							return Err(ApplicationError::StartupError(String::from(
								"棋譜ファイルのパースでエラーが発生しました。"
							)));
						}
					};

					let mut state = State::new(banmen);

					let len = if mvs.len() < fromlast as usize {
						mvs.len()
					} else {
						mvs.len() - fromlast as usize
					};

					let mvs = mvs.into_iter().take(len).collect::<Vec<Move>>();

					for m in &mvs {
						match Rule::apply_move_none_check(&state,teban,&mc,m.to_applied_move()) {
							(s,nmc,_) => {
								state = s;
								mc = nmc;
								teban = teban.opposite();
							}
						}
					}

					sfen_list.push((teban, state.get_banmen().clone(), mc, Vec::new()).to_sfen()?);

					buf.clear();
				}

				let mut rnd = rand::thread_rng();
				let mut rnd = XorShiftRng::from_seed(rnd.gen());
				let len = sfen_list.len();

				let f:Box<FnMut() -> String + Send + 'static> = Box::new(move || {
					sfen_list[rnd.gen_range(0, len)].clone()
				});

				Some(f)
			}
		};

		print!("base_depth = {:?}, max_depth = {:?}, time_limit = {:?}, running_time = {:?}, number_of_games = {:?}",
			base_depth, max_depth, time_limit, running_time_none_parsed, number_of_games
		);

		let info_sender_arc = Arc::new(Mutex::new(CosoleInfoSender::new(silent)));

		let mut engine = SelfMatchEngine::new(
			NNShogiPlayer::new(String::from("nn.a.bin"),String::from("nn.b.bin"),true),
			NNShogiPlayer::new(String::from("nn_opponent.a.bin"),String::from("nn_opponent.b.bin"),true),
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
								initial_position_creator,Some(FileSfenKifuWriter::new(String::from("logs/kifu.txt"))?),
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
		)))).map(|r| {
			print!("開始日時: {}\n",r.start_dt.format("%Y年%m月%d日 %H:%M:%S").to_string());
			print!("終了日時: {}\n",r.end_dt.format("%Y年%m月%d日 %H:%M:%S").to_string());
			print!("試合回数: {}\n",r.game_count);
			let secs = r.elapsed.as_secs();
			print!("経過時間: {}時間{}分{}.{:?}秒\n",
					secs / (60 * 60), secs  % (60 * 60) / 60, secs % 60, r.elapsed.subsec_nanos() / 1_000_000);
		})
	} else {
		let agent = UsiAgent::new(NNShogiPlayer::new(String::from("nn.a.bin"),String::from("nn.b.bin"),false));

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
				print!("{}\n",command.to_usi_command()?);
			}
		}
		Ok(())
	}
}