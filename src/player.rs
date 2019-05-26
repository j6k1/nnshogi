use std::collections::HashMap;
use std::fmt;
use rand;
use rand::Rng;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::sync::Arc;
use std::sync::Mutex;
use error::*;
use std::num::Wrapping;
use std::time::{Instant,Duration};
use std::cmp::Ordering;
use std::ops::Neg;

use usiagent::player::*;
use usiagent::event::*;
use usiagent::command::*;
use usiagent::shogi::*;
use usiagent::rule::*;
use usiagent::hash::*;
use usiagent::OnErrorHandler;
use usiagent::logger::*;
use usiagent::error::PlayerError;
use usiagent::error::UsiProtocolError;
use usiagent::TryFrom;

use nn::Intelligence;

const KOMA_KIND_MAX:usize = KomaKind::Blank as usize;
const MOCHIGOMA_KIND_MAX:usize = MochigomaKind::Hisha as usize;
const MOCHIGOMA_MAX:usize = 18;
const SUJI_MAX:usize = 9;
const DAN_MAX:usize = 9;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Evaluation {
	Result(Score,Option<Move>),
	Timeout(Option<Move>),
	Error,
}
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum OuteEvaluation {
	Result(i32),
	Timeout,
}
#[derive(Clone, Copy, PartialEq, Debug)]
enum Score {
	NEGINFINITE,
	Value(f64),
	NegativeValue(f64),
	INFINITE,
}
impl Neg for Score {
	type Output = Score;

	fn neg(self) -> Score {
		match self {
			Score::INFINITE => Score::NEGINFINITE,
			Score::NEGINFINITE => Score::INFINITE,
			Score::Value(v) => Score::NegativeValue(v),
			Score::NegativeValue(v) => Score::Value(v),
		}
	}
}
impl PartialOrd for Score {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering>	{
		Some(match *self {
			Score::INFINITE if *other == Score::INFINITE => {
				Ordering::Equal
			},
			Score::INFINITE => Ordering::Greater,
			Score::NEGINFINITE if *other == Score::NEGINFINITE => {
				Ordering::Equal
			},
			Score::NEGINFINITE => Ordering::Less,
			Score::Value(l) => {
				match *other {
					Score::Value(r) => l.partial_cmp(&r)?,
					Score::NegativeValue(r) => r.partial_cmp(&l)?,
					Score::INFINITE => Ordering::Less,
					Score::NEGINFINITE => Ordering::Greater,
				}
			},
			Score::NegativeValue(l) => {
				match *other {
					Score::NegativeValue(r) => r.partial_cmp(&l)?,
					Score::Value(r) => l.partial_cmp(&r)?,
					Score::INFINITE => Ordering::Less,
					Score::NEGINFINITE => Ordering::Greater,
				}
			}
		})
	}
}
const BASE_DEPTH:u32 = 2;
const MAX_DEPTH:u32 = 6;
const TIMELIMIT_MARGIN:u64 = 50;
const NETWORK_DELAY:u32 = 1100;
const DEFALUT_DISPLAY_EVALUTE_SCORE:bool = false;

pub struct NNShogiPlayer {
	stop:bool,
	pub quited:bool,
	kyokumen_hash_seeds:[[u64; SUJI_MAX * DAN_MAX]; KOMA_KIND_MAX + 1],
	mochigoma_hash_seeds:[[[u64; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX]; 2],
	kyokumen:Option<Arc<Kyokumen>>,
	mhash:u64,
	shash:u64,
	oute_kyokumen_hash_map:TwoKeyHashMap<u64,()>,
	kyokumen_hash_map:TwoKeyHashMap<u64,u32>,
	nna_filename:String,
	nnb_filename:String,
	learning_mode:bool,
	evalutor:Option<Intelligence>,
	pub history:Vec<(Banmen,MochigomaCollections,u64,u64)>,
	base_depth:u32,
	max_depth:u32,
	network_delay:u32,
	display_evalute_score:bool,
	count_of_move_started:u32,
	moved:bool,
}
impl fmt::Debug for NNShogiPlayer {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "NNShogiPlayer")
	}
}
impl NNShogiPlayer {
	pub fn new(nna_filename:String,nnb_filename:String,learning_mode:bool) -> NNShogiPlayer {
		let mut rnd = rand::thread_rng();
		let mut rnd = XorShiftRng::from_seed(rnd.gen());

		let mut kyokumen_hash_seeds:[[u64; SUJI_MAX * DAN_MAX]; KOMA_KIND_MAX + 1] = [[0; SUJI_MAX * DAN_MAX]; KOMA_KIND_MAX + 1];
		let mut mochigoma_hash_seeds:[[[u64; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX]; 2] = [[[0; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX]; 2];

		for i in 0..(KOMA_KIND_MAX + 1) {
			for j in 0..(SUJI_MAX * DAN_MAX) {
				kyokumen_hash_seeds[i][j] = rnd.gen();
			}
		}

		for i in 0..MOCHIGOMA_MAX {
			for j in 0..(MOCHIGOMA_KIND_MAX + 1) {
				mochigoma_hash_seeds[0][i][j] = rnd.gen();
				mochigoma_hash_seeds[1][i][j] = rnd.gen();
			}
		}

		NNShogiPlayer {
			stop:false,
			quited:false,
			kyokumen_hash_seeds:kyokumen_hash_seeds,
			mochigoma_hash_seeds:mochigoma_hash_seeds,
			kyokumen:None,
			mhash:0,
			shash:0,
			oute_kyokumen_hash_map:TwoKeyHashMap::new(),
			kyokumen_hash_map:TwoKeyHashMap::new(),
			nna_filename:nna_filename,
			nnb_filename:nnb_filename,
			learning_mode:learning_mode,
			evalutor:None,
			history:Vec::new(),
			base_depth:BASE_DEPTH,
			max_depth:MAX_DEPTH,
			network_delay:NETWORK_DELAY,
			display_evalute_score:DEFALUT_DISPLAY_EVALUTE_SCORE,
			count_of_move_started:0,
			moved:false,
		}
	}

	fn timelimit_reached(&self,limit:&Option<Instant>) -> bool {
		let network_delay = self.network_delay;
		limit.map_or(false,|l| {
			l < Instant::now() || l - Instant::now() <= Duration::from_millis(network_delay as u64 + TIMELIMIT_MARGIN)
		})
	}

	fn send_message<L,S>(&mut self, info_sender:&mut S,
			on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>, message:&str)
		where L: Logger, S: InfoSender,
			Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
		commands.push(UsiInfoSubCommand::Str(String::from(message)));

		match info_sender.send(commands) {
			Ok(_) => (),
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}
	}

	fn send_seldepth<L,S>(&mut self, info_sender:&mut S,
			on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>, depth:u32, seldepth:u32)
		where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
		commands.push(UsiInfoSubCommand::Depth(depth));
		commands.push(UsiInfoSubCommand::SelDepth(seldepth));


		match info_sender.send(commands) {
			Ok(_) => (),
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}
	}
	/*
	fn send_depth<L>(&mut self, info_sender:&USIInfoSender,
			on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>, depth:u32)
		where L: Logger {
		let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
		commands.push(UsiInfoSubCommand::Depth(depth));

		match info_sender.send(commands) {
			Ok(_) => (),
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}
	}
	*/

	fn evalute<L,S>(&mut self,teban:Teban,state:&State,mc:&MochigomaCollections,m:&Option<Move>,
					info_sender:&mut S,on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>)
		-> Evaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let s = match self.evalutor {
			Some(ref mut evalutor) => {
				match evalutor.evalute(teban,state.get_banmen(),mc) {
					Ok(s) => Some(s),
					Err(ref mut e) => {
						on_error_handler.lock().map(|h| h.call(e)).is_err();
						return Evaluation::Error;
					}
				}
			},
			None => None,
		};

		match s {
			Some(s) => {
				if self.display_evalute_score {
					self.send_message(info_sender, on_error_handler, &format!("evalute score = {}",s));
				}
				return Evaluation::Result(Score::Value(s),m.clone());
			},
			None => {
				self.send_message(info_sender, on_error_handler, &format!("evalutor is not initialized!"));
				return Evaluation::Error;
			}
		}
	}

	fn alphabeta<'a,L,S>(&mut self,
			event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>,
								info_sender:&mut S,
								on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
								teban:Teban,state:&State,
								mut alpha:Score,beta:Score,
								m:Option<Move>,mc:&MochigomaCollections,
								obtained:Option<ObtainKind>,
								current_kyokumen_hash_map:&TwoKeyHashMap<u64,u32>,
								already_oute_hash_map:&mut TwoKeyHashMap<u64,()>,
								ignore_oute_hash_map:&mut TwoKeyHashMap<u64,()>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,
								depth:u32,current_depth:u32,base_depth:u32)
		-> Evaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		if current_depth > base_depth {
			self.send_seldepth(info_sender, on_error_handler, base_depth, current_depth);
		}

		match obtained {
			Some(ObtainKind::Ou) => {
				return Evaluation::Result(Score::NEGINFINITE,None);
			},
			_ => (),
		}

		match self.handle_events(event_queue, on_error_handler) {
			Ok(_) => (),
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}

		if self.timelimit_reached(&limit) || self.stop {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return Evaluation::Timeout(None);
		}

		let (mvs,responded_oute) = if Rule::is_mate(teban.opposite(),state) {
			if depth == 0 || current_depth == self.max_depth {
				if self.timelimit_reached(&limit) || self.stop {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return Evaluation::Timeout(None);
				} else {
					return self.evalute(teban,state,mc,&m,info_sender,on_error_handler);
				}
			}

			(Rule::respond_oute_only_moves_all(teban, state, mc),true)
		} else {
			let oute_mvs = Rule::oute_only_moves_all(teban,&state,mc);

			if oute_mvs.len() > 0 {
				match oute_mvs[0] {
					LegalMove::To(ref m) if m.obtained() == Some(ObtainKind::Ou) => {
						return Evaluation::Result(Score::INFINITE,Some(oute_mvs[0].to_move()));
					},
					_ => (),
				}
			}

			match self.handle_events(event_queue, on_error_handler) {
				Ok(_) => (),
				Err(ref e) => {
					on_error_handler.lock().map(|h| h.call(e)).is_err();
				}
			}

			if self.timelimit_reached(&limit) || self.stop {
				self.send_message(info_sender, on_error_handler, "think timeout!");
				return Evaluation::Timeout(Some(oute_mvs[0].to_move()))
			}

			if depth == 0 || current_depth == self.max_depth {
				if self.timelimit_reached(&limit) || self.stop {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return Evaluation::Timeout(None);
				} else {
					return self.evalute(teban,state,mc,&m,info_sender,on_error_handler);
				}
			}

			for m in &oute_mvs {
				let o = match m {
					LegalMove::To(ref m) => m.obtained().and_then(|o| MochigomaKind::try_from(o).ok()),
					_ => None,
				};

				let mhash = self.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
				let shash = self.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

				match self.oute_kyokumen_hash_map.get(&mhash,&shash) {
					Some(_) => {
						continue;
					},
					None => (),
				}

				let mut current_kyokumen_hash_map = current_kyokumen_hash_map.clone();

				match current_kyokumen_hash_map.get(&mhash,&shash) {
					Some(c) if c >= 3 => {
						continue;
					},
					Some(c) => {
						current_kyokumen_hash_map.insert(mhash,shash,c+1);
					},
					None => {
						current_kyokumen_hash_map.insert(mhash,shash,1);
					}
				}

				match already_oute_hash_map.get(&mhash,&shash) {
					None => {
						already_oute_hash_map.insert(mhash,shash,());
					},
					Some(_) => (),
				}

				match ignore_oute_hash_map.get(&mhash,&shash) {
					Some(_) => {
						continue;
					},
					_ => (),
				}

				let next = Rule::apply_move_none_check(&state,teban,mc,m.to_applied_move());

				match next {
					(ref next,ref mc,_) if !Rule::is_mate(teban.opposite(),next) => {
						let is_put_fu = match m {
							LegalMove::Put(ref m) if m.kind() == MochigomaKind::Fu => true,
							_ => false,
						};

						match self.handle_events(event_queue, on_error_handler) {
							Ok(_) => (),
							Err(ref e) => {
								on_error_handler.lock().map(|h| h.call(e)).is_err();
							}
						}

						if self.timelimit_reached(&limit) || self.stop {
							self.send_message(info_sender, on_error_handler, "think timeout!");
							return Evaluation::Timeout(Some(m.to_move()));
						}

						match self.respond_oute_only(event_queue,
															info_sender,
															on_error_handler,
															teban.opposite(),next,mc,
															&current_kyokumen_hash_map,
															already_oute_hash_map,
															ignore_oute_hash_map,
															mhash,shash,limit,
															current_depth+1,
															base_depth) {
							OuteEvaluation::Result(d) if d >= 0 &&
								!(is_put_fu && d - current_depth as i32 == 2) => {
								return Evaluation::Result(Score::INFINITE,Some(m.to_move()));
							},
							OuteEvaluation::Timeout => {
								return Evaluation::Timeout(Some(m.to_move()));
							},
							_ => (),
						}
					},
					_ => (),
				}

				match self.handle_events(event_queue, on_error_handler) {
					Ok(_) => (),
					Err(ref e) => {
						on_error_handler.lock().map(|h| h.call(e)).is_err();
					}
				}

				if self.timelimit_reached(&limit) || self.stop {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return Evaluation::Timeout(Some(m.to_move()));
				}
			}

			if oute_mvs.len() == 0 {
				match self.handle_events(event_queue, on_error_handler) {
					Ok(_) => (),
					Err(ref e) => {
						on_error_handler.lock().map(|h| h.call(e)).is_err();
					}
				}

				if self.timelimit_reached(&limit) || self.stop {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return Evaluation::Timeout(None);
				}
			} else {
				if self.timelimit_reached(&limit) || self.stop {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return Evaluation::Timeout(Some(oute_mvs[0].to_move()));
				}
			}

			let mvs:Vec<LegalMove> = Rule::legal_moves_all(teban, &state, mc);

			(mvs,false)
		};

		if mvs.len() == 0 {
			return Evaluation::Result(Score::NEGINFINITE,None);
		} else if mvs.len() == 1 {
			return self.evalute(teban,state,mc,&Some(mvs[0].to_move()),info_sender,on_error_handler);
		}

		match self.handle_events(event_queue, on_error_handler) {
			Ok(_) => (),
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}

		if self.timelimit_reached(&limit) || self.stop {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return Evaluation::Timeout(Some(mvs[0].to_move()));
		}

		let mut scoreval = Score::NEGINFINITE;
		let mut best_move:Option<Move> = None;

		for m in &mvs {
			let obtained = match m {
				LegalMove::To(ref m) => m.obtained(),
				_ => None,
			};

			let (mhash,shash) = {
				let o = match obtained {
					Some(o) => {
						match MochigomaKind::try_from(o) {
							Ok(o) => {
								Some(o)
							},
							Err(_) => None,
						}
					},
					None => None,
				};

				let mhash = self.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
				let shash = self.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

				match self.oute_kyokumen_hash_map.get(&mhash,&shash) {
					Some(_) => {
						continue;
					},
					None => (),
				}
				(mhash,shash)
			};

			let m = m.to_applied_move();
			let next = Rule::apply_move_none_check(&state,teban,mc,m);

			let depth = match obtained {
				Some(_) => depth + 1,
				None => {
					if let AppliedMove::To(ref mv) = m {
						let ps = Rule::apply_move_to_partial_state_none_check(state,teban,mc,m);
						let banmen = state.get_banmen();
						let (x,y) = mv.src().square_to_point();

						if Rule::is_mate_with_partial_state_and_point_and_kind(teban,&ps,x,y,banmen.0[y as usize][x as usize]) ||
						   Rule::is_mate_with_partial_state_repeat_move_kinds(teban,&ps) {
							depth + 1
						} else {
							depth
						}
					} else if responded_oute {
						depth + 1
					} else {
						depth
					}
				}
			};

			match next {
				(ref state,ref mc,_) => {

					let mut current_kyokumen_hash_map = current_kyokumen_hash_map.clone();

					match current_kyokumen_hash_map.get(&mhash,&shash) {
						Some(c) if c >= 3 => {
							continue;
						},
						Some(c) => {
							current_kyokumen_hash_map.insert(mhash,shash,c+1);

							let s = if Rule::is_mate(teban.opposite(),state) {
								Score::NEGINFINITE
							} else {
								Score::Value(0f64)
							};

							if s > scoreval {
								scoreval = s;
								best_move = Some(m.to_move());
								if alpha < scoreval {
									alpha = scoreval;
								}
								if scoreval >= beta {
									return Evaluation::Result(scoreval,best_move);
								}
							}

							continue;
						},
						None => {
							current_kyokumen_hash_map.insert(mhash,shash,1);
						}
					}

					match self.alphabeta(event_queue,
						info_sender,
						on_error_handler,
						teban.opposite(),&state,
						-beta,-alpha,Some(m.to_move()),&mc,
						obtained,&current_kyokumen_hash_map,
						already_oute_hash_map,
						ignore_oute_hash_map,
						mhash,shash,limit,depth-1,
						current_depth+1,base_depth) {

						Evaluation::Timeout(_) => {
							return match best_move {
								Some(best_move) => Evaluation::Timeout(Some(best_move)),
								None => Evaluation::Timeout(Some(m.to_move())),
							};
						},
						Evaluation::Result(s,_) => {
							if -s > scoreval {
								scoreval = -s;
								best_move = Some(m.to_move());
								if alpha < scoreval {
									alpha = scoreval;
								}
								if scoreval >= beta {
									return Evaluation::Result(scoreval,best_move);
								}
							}
						},
						Evaluation::Error => {
							return Evaluation::Error
						}
					}
				}
			}

			match self.handle_events(event_queue, on_error_handler) {
				Ok(_) => (),
				Err(ref e) => {
					on_error_handler.lock().map(|h| h.call(e)).is_err();
				}
			}

			if self.timelimit_reached(&limit) || self.stop {
				self.send_message(info_sender, on_error_handler, "think timeout!");
				return match best_move {
					Some(best_move) => Evaluation::Timeout(Some(best_move)),
					None => Evaluation::Timeout(Some(m.to_move())),
				};
			}
		}

		Evaluation::Result(scoreval,best_move)
	}

	fn respond_oute_only<'a,L,S>(&mut self,
								event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>,
								info_sender:&mut S,
								on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
								teban:Teban,state:&State,
								mc:&MochigomaCollections,
								current_kyokumen_hash_map:&TwoKeyHashMap<u64,u32>,
								already_oute_hash_map:&mut TwoKeyHashMap<u64,()>,
								ignore_oute_hash_map:&mut TwoKeyHashMap<u64,()>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,
								current_depth:u32,
								base_depth:u32)
		-> OuteEvaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let mvs = Rule::respond_oute_only_moves_all(teban,&state, mc);

		//self.send_seldepth(info_sender, on_error_handler, base_depth, current_depth);

		match self.handle_events(event_queue, on_error_handler) {
			Ok(_) => (),
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}

		if self.timelimit_reached(&limit) || self.stop {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return OuteEvaluation::Timeout;
		}

		if mvs.len() == 0 {
			return OuteEvaluation::Result(current_depth as i32)
		} else {
			let mut maxdepth = -1;

			for m in &mvs {
				let o = match m {
					&LegalMove::To(ref m) => {
						match m.obtained() {
							Some(o) => {
								match MochigomaKind::try_from(o) {
									Ok(o) => Some(o),
									Err(_) => None,
								}
							},
							None => None,
						}
					},
					_ => None,
				};
				let mhash = self.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
				let shash = self.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

				match already_oute_hash_map.get(&mhash,&shash) {
					None => {
						already_oute_hash_map.insert(mhash,shash,());
					},
					Some(_) => {
						return OuteEvaluation::Result(-1);
					}
				}

				let mut current_kyokumen_hash_map = current_kyokumen_hash_map.clone();

				match current_kyokumen_hash_map.get(&mhash,&shash) {
					Some(c) if c >= 3 => {
						continue;
					},
					Some(c) => {
						current_kyokumen_hash_map.insert(mhash,shash,c+1);
					},
					None => {
						current_kyokumen_hash_map.insert(mhash,shash,1);
					}
				}

				let next = Rule::apply_move_none_check(&state,teban,mc,m.to_applied_move());

				match next {
					(ref next,ref mc,_) => {
						match self.oute_only(event_queue,
												info_sender,
												on_error_handler,
												teban.opposite(),next,mc,
												&current_kyokumen_hash_map,
												already_oute_hash_map,
												ignore_oute_hash_map,
												mhash,shash,limit,
												current_depth+1,base_depth) {
							OuteEvaluation::Result(-1) => {
								return OuteEvaluation::Result(-1);
							},
							OuteEvaluation::Result(d) => {
								if d > maxdepth {
									maxdepth = d;
								}
							},
							OuteEvaluation::Timeout => {
								return OuteEvaluation::Timeout;
							},
						}
					}
				}

				match self.handle_events(event_queue, on_error_handler) {
					Ok(_) => (),
					Err(ref e) => {
						on_error_handler.lock().map(|h| h.call(e)).is_err();
					}
				}
				if self.timelimit_reached(&limit) || self.stop {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return OuteEvaluation::Timeout;
				}
			}

			OuteEvaluation::Result(maxdepth)
		}
	}

	fn oute_only<'a,L,S>(&mut self,
								event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>,
								info_sender:&mut S,
								on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
								teban:Teban,state:&State,
								mc:&MochigomaCollections,
								current_kyokumen_hash_map:&TwoKeyHashMap<u64,u32>,
								already_oute_hash_map:&mut TwoKeyHashMap<u64,()>,
								ignore_oute_hash_map:&mut TwoKeyHashMap<u64,()>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,
								current_depth:u32,
								base_depth:u32)
		-> OuteEvaluation where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let mvs = Rule::oute_only_moves_all(teban, &state, mc);

		//self.send_seldepth(info_sender, on_error_handler, base_depth, current_depth);

		match self.handle_events(event_queue, on_error_handler) {
			Ok(_) => (),
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}
		if self.timelimit_reached(&limit) || self.stop {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return OuteEvaluation::Timeout;
		}

		if mvs.len() == 0 {
			OuteEvaluation::Result(-1)
		} else {
			for m in &mvs {
				match m {
					&LegalMove::To(ref m) => {
						if let Some(ObtainKind::Ou) = m.obtained() {
							return OuteEvaluation::Result(current_depth as i32);
						}
					},
					_ => ()
				}

				let is_put_fu = match m {
					&LegalMove::Put(ref m) if m.kind() == MochigomaKind::Fu => true,
					_ => false,
				};

				let next = Rule::apply_move_none_check(&state,teban,mc,m.to_applied_move());

				let o = match m {
					&LegalMove::To(ref m) => {
						match m.obtained() {
							Some(o) => {
								match MochigomaKind::try_from(o) {
									Ok(o) => Some(o),
									Err(_) => None,
								}
							},
							None => None,
						}
					},
					_ => None,
				};

				let mhash = self.calc_main_hash(mhash,&teban,state.get_banmen(),mc,&m.to_move(),&o);
				let shash = self.calc_sub_hash(shash,&teban,state.get_banmen(),mc,&m.to_move(),&o);

				match already_oute_hash_map.get(&mhash,&shash) {
					None => {
						already_oute_hash_map.insert(mhash,shash,());
					},
					Some(_) => {
						return OuteEvaluation::Result(-1);
					}
				}

				let mut current_kyokumen_hash_map = current_kyokumen_hash_map.clone();

				match current_kyokumen_hash_map.get(&mhash,&shash) {
					Some(c) if c >= 3 => {
						continue;
					},
					Some(c) => {
						current_kyokumen_hash_map.insert(mhash,shash,c+1);
					},
					None => {
						current_kyokumen_hash_map.insert(mhash,shash,1);
					}
				}

				match next {
					(ref next,ref mc,_) => {
						match self.respond_oute_only(event_queue,
														info_sender,
														on_error_handler,
														teban.opposite(),next,mc,
														&current_kyokumen_hash_map,
														already_oute_hash_map,
														ignore_oute_hash_map,
														mhash,shash,limit,
														current_depth+1,base_depth) {
							OuteEvaluation::Result(-1) => {
								return OuteEvaluation::Result(-1);
							}
							OuteEvaluation::Result(d) if d >= 0 &&
														!(is_put_fu && d - current_depth as i32 == 2)=> {
								return OuteEvaluation::Result(d);
							},
							OuteEvaluation::Timeout => {
								return OuteEvaluation::Timeout;
							},
							_ => (),
						}
					}
				}

				match self.handle_events(event_queue, on_error_handler) {
					Ok(_) => (),
					Err(ref e) => {
						on_error_handler.lock().map(|h| h.call(e)).is_err();
					}
				}

				if self.timelimit_reached(&limit) || self.stop {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return OuteEvaluation::Timeout;
				}
			}
			OuteEvaluation::Result(-1)
		}
	}

	fn calc_hash<AF,PF>(&self,h:u64,t:&Teban,b:&Banmen,mc:&MochigomaCollections,
												m:&Move,obtained:&Option<MochigomaKind>,add:AF,pull:PF)
		-> u64 where AF: Fn(u64,u64) -> u64, PF: Fn(u64,u64) -> u64 {
		match b {
			&Banmen(ref kinds) => {
				match m {
					&Move::To(KomaSrcPosition(sx,sy), KomaDstToPosition(dx, dy, n)) => {
						let sx = (9 - sx) as usize;
						let sy = (sy - 1) as usize;
						let dx = (9 - dx) as usize;
						let dy = dy as usize - 1;

						let mut hash = h;
						let k = kinds[sy][sx];

						hash =  pull(hash,self.kyokumen_hash_seeds[k as usize][sy * 8 + sx]);
						hash = add(hash,self.kyokumen_hash_seeds[KomaKind::Blank as usize][sy * 8 + sx]);

						let dk = kinds[dy][dx] as usize;

						hash =  pull(hash,self.kyokumen_hash_seeds[dk][dy * 8 + dx]);

						let k = if n {
							match k {
								KomaKind::SFu => KomaKind::SFuN,
								KomaKind::SKyou => KomaKind::SKyouN,
								KomaKind::SKei => KomaKind::SKeiN,
								KomaKind::SGin => KomaKind::SGinN,
								KomaKind::SKaku => KomaKind::SKakuN,
								KomaKind::SHisha => KomaKind::SHishaN,
								KomaKind::GFu => KomaKind::GFuN,
								KomaKind::GKyou => KomaKind::GKyouN,
								KomaKind::GKei => KomaKind::GKeiN,
								KomaKind::GGin => KomaKind::GGinN,
								KomaKind::GKaku => KomaKind::GKakuN,
								KomaKind::GHisha => KomaKind::GHishaN,
								k => k,
							}
						} else {
							k
						} as usize;

						hash = add(hash,self.kyokumen_hash_seeds[k][dy * 8 + dx]);

						hash = match obtained  {
								&None => hash,
								&Some(ref obtained) => {
									let c =  match t {
										&Teban::Sente => {
											match mc {
												&MochigomaCollections::Pair(ref mc,_) => {
													match mc.get(obtained) {
														Some(c) => *c as usize,
														None => 0,
													}
												},
												&MochigomaCollections::Empty => 0,
											}
										},
										&Teban::Gote => {
											match mc {
												&MochigomaCollections::Pair(_,ref mc) => {
													match mc.get(obtained) {
														Some(c) => *c as usize,
														None => 0,
													}
												},
												&MochigomaCollections::Empty => 0,
											}
										}
									};

									let k = *obtained as usize;

									match t {
										&Teban::Sente => {
											hash = add(hash,self.mochigoma_hash_seeds[0][c][k]);
										},
										&Teban::Gote => {
											hash = add(hash,self.mochigoma_hash_seeds[1][c][k]);
										}
									}
									hash
								}
						};

						hash
					},
					&Move::Put(ref mk, ref md) => {
						let mut hash = h;

						let c = match t {
							&Teban::Sente => {
								match mc {
									&MochigomaCollections::Pair(ref mc,_) => {
										match mc.get(&mk) {
											None | Some(&0) => {
												return hash;
											}
											Some(c) => *c as usize,
										}
									},
									&MochigomaCollections::Empty => {
										return hash;
									}
								}
							},
							&Teban::Gote => {
								match mc {
									&MochigomaCollections::Pair(_,ref mc) => {
										match mc.get(&mk) {
											None | Some(&0) => {
												return hash;
											}
											Some(c) => *c as usize,
										}
									},
									&MochigomaCollections::Empty => {
										return hash;
									}
								}
							}
						};

						let k = *mk as usize;

						match t {
							&Teban::Sente => {
								hash = pull(hash,self.mochigoma_hash_seeds[0][c-1][k]);
							},
							&Teban::Gote => {
								hash = pull(hash,self.mochigoma_hash_seeds[1][c-1][k]);
							}
						}

						let dx = 9 - md.0 as usize;
						let dy = md.1 as usize - 1;
						let dk = kinds[dy][dx] as usize;

						hash = pull(hash,self.kyokumen_hash_seeds[dk as usize][dy * 8 + dx]);

						let k = KomaKind::from((*t,*mk)) as usize;

						hash = add(hash,self.kyokumen_hash_seeds[k as usize][dy * 8 + dx]);
						hash
					}
				}
			}
		}
	}

	fn calc_main_hash(&self,h:u64,t:&Teban,b:&Banmen,mc:&MochigomaCollections,m:&Move,obtained:&Option<MochigomaKind>) -> u64 {
		self.calc_hash(h,t,b,mc,m,obtained,|h,v| h ^ v, |h,v| h ^ v)
	}

	fn calc_sub_hash(&self,h:u64,t:&Teban,b:&Banmen,mc:&MochigomaCollections,m:&Move,obtained:&Option<MochigomaKind>) -> u64 {
		self.calc_hash(h,t,b,mc,m,obtained,|h,v| {
			let h = Wrapping(h);
			let v = Wrapping(v);
			(h + v).0
		}, |h,v| {
			let h = Wrapping(h);
			let v = Wrapping(v);
			(h - v).0
		})
	}

	fn calc_initial_hash(&self,b:&Banmen,
		ms:&HashMap<MochigomaKind,u32>,mg:&HashMap<MochigomaKind,u32>) -> (u64,u64) {
		let mut mhash:u64 = 0;
		let mut shash:Wrapping<u64> = Wrapping(0u64);

		match b {
			&Banmen(ref kinds) => {
				for y in 0..9 {
					for x in 0..9 {
						let k = kinds[y][x] as usize;
						mhash = mhash ^ self.kyokumen_hash_seeds[k][y * 8 + x];
						shash = shash + Wrapping(self.kyokumen_hash_seeds[k][y * 8 + x]);
					}
				}
			}
		}
		for k in &MOCHIGOMA_KINDS {
			match ms.get(&k) {
				Some(c) => {
					for i in 0..(*c as usize) {
						mhash = mhash ^ self.mochigoma_hash_seeds[0][i][*k as usize];
						shash = shash + Wrapping(self.mochigoma_hash_seeds[0][i][*k as usize]);
					}
				},
				None => (),
			}
			match mg.get(&k) {
				Some(c) => {
					for i in 0..(*c as usize) {
						mhash = mhash ^ self.mochigoma_hash_seeds[1][i][*k as usize];
						shash = shash + Wrapping(self.mochigoma_hash_seeds[1][i][*k as usize]);
					}
				},
				None => (),
			}
		}

		(mhash,shash.0)
	}
}
impl USIPlayer<CommonError> for NNShogiPlayer {
	const ID: &'static str = "nnshogi";
	const AUTHOR: &'static str = "jinpu";
	fn get_option_kinds(&mut self) -> Result<HashMap<String,SysEventOptionKind>,CommonError> {
		let mut kinds:HashMap<String,SysEventOptionKind> = HashMap::new();
		kinds.insert(String::from("USI_Hash"),SysEventOptionKind::Num);
		kinds.insert(String::from("USI_Ponder"),SysEventOptionKind::Bool);
		kinds.insert(String::from("MaxDepth"),SysEventOptionKind::Num);
		kinds.insert(String::from("BaseDepth"),SysEventOptionKind::Num);
		kinds.insert(String::from("NetworkDelay"),SysEventOptionKind::Num);
		kinds.insert(String::from("DispEvaluteScore"),SysEventOptionKind::Bool);

		Ok(kinds)
	}
	fn get_options(&mut self) -> Result<HashMap<String,UsiOptType>,CommonError> {
		let mut options:HashMap<String,UsiOptType> = HashMap::new();
		options.insert(String::from("BaseDepth"),UsiOptType::Spin(1,100,Some(BASE_DEPTH)));
		options.insert(String::from("MaxDepth"),UsiOptType::Spin(1,100,Some(MAX_DEPTH)));
		options.insert(String::from("NetworkDelay"),UsiOptType::Spin(0,10000,Some(NETWORK_DELAY)));
		options.insert(String::from("DispEvaluteScore"),UsiOptType::Check(Some(DEFALUT_DISPLAY_EVALUTE_SCORE)));
		Ok(options)
	}
	fn take_ready(&mut self) -> Result<(),CommonError> {
		match self.evalutor {
			Some(_) => (),
			None => {
				self.evalutor = Some(Intelligence::new(
										String::from("data"),
										self.nna_filename.clone(),
										self.nnb_filename.clone(),self.learning_mode));
			}
		}
		Ok(())
	}
	fn set_option(&mut self,name:String,value:SysEventOption) -> Result<(),CommonError> {
		match &*name {
			"MaxDepth" => {
				self.max_depth = match value {
					SysEventOption::Num(depth) => {
						depth
					},
					_ => MAX_DEPTH,
				};
			},
			"BaseDepth" => {
				self.base_depth = match value {
					SysEventOption::Num(depth) => {
						depth
					},
					_ => BASE_DEPTH,
				};
			},
			"NetworkDelay" => {
				self.network_delay = match value {
					SysEventOption::Num(n) => {
						n
					},
					_ => NETWORK_DELAY,
				}
			},
			"DispEvaluteScore" => {
				self.display_evalute_score =  match value {
					SysEventOption::Bool(b) => {
						b
					},
					_ => DEFALUT_DISPLAY_EVALUTE_SCORE,
				}
			}
			_ => (),
		}
		Ok(())
	}
	fn newgame(&mut self) -> Result<(),CommonError> {
		self.kyokumen = None;
		self.history.clear();
		if !self.quited {
			self.stop = false;
		}
		self.count_of_move_started = 0;
		Ok(())
	}
	fn set_position(&mut self,teban:Teban,banmen:Banmen,
					ms:HashMap<MochigomaKind,u32>,mg:HashMap<MochigomaKind,u32>,_:u32,m:Vec<Move>)
		-> Result<(),CommonError> {
		self.history.clear();
		self.kyokumen_hash_map.clear();

		let mut kyokumen_hash_map:TwoKeyHashMap<u64,u32> = TwoKeyHashMap::new();
		let (mhash,shash) = self.calc_initial_hash(&banmen,&ms,&mg);

		kyokumen_hash_map.insert(mhash,shash,1);

		let teban = teban;
		let state = State::new(banmen);

		let mc = MochigomaCollections::new(ms,mg);


		let history:Vec<(Banmen,MochigomaCollections,u64,u64)> = Vec::new();

		let (t,state,mc,r) = self.apply_moves(teban,state,
												mc,m.into_iter()
													.map(|m| m.to_applied_move())
													.collect::<Vec<AppliedMove>>(),
												(mhash,shash,kyokumen_hash_map,history),
												|s,t,banmen,mc,m,o,r| {
			let (prev_mhash,prev_shash,mut kyokumen_hash_map,mut history) = r;

			let (mhash,shash) = match m {
				&Some(ref m) => {
					let mhash = s.calc_main_hash(prev_mhash,&t,&banmen,&mc,&m.to_move(),&o);
					let shash = s.calc_sub_hash(prev_shash,&t,&banmen,&mc,&m.to_move(),&o);

					match kyokumen_hash_map.get(&mhash,&shash) {
						Some(c) => {
							kyokumen_hash_map.insert(mhash,shash,c+1);
						},
						None => {
							kyokumen_hash_map.insert(mhash,shash,1);
						}
					};
					(mhash,shash)
				},
				&None => {
					(prev_mhash,prev_shash)
				}
			};

			history.push((banmen.clone(),mc.clone(),prev_mhash,prev_shash));
			(mhash,shash,kyokumen_hash_map,history)
		});

		let (mhash,shash,kyokumen_hash_map,history) = r;

		let mut oute_kyokumen_hash_map:TwoKeyHashMap<u64,()> = TwoKeyHashMap::new();
		let mut current_teban = t.opposite();
		let opponent = t.opposite();

		for h in history.iter().rev().skip(1) {
			if current_teban == opponent {
				match &h {
					&(ref banmen,_, mhash,shash) => {
						if Rule::win_only_moves(t,&State::new(banmen.clone())).len() == 0 {
							break;
						} else {
							oute_kyokumen_hash_map.insert(*mhash,*shash,());
						}
					}
				}
			}

			current_teban = current_teban.opposite();
		}

		self.kyokumen = Some(Arc::new(Kyokumen {
			state:state,
			mc:mc,
			teban:t
		}));
		self.mhash = mhash;
		self.shash = shash;
		self.oute_kyokumen_hash_map = oute_kyokumen_hash_map;
		self.kyokumen_hash_map = kyokumen_hash_map;
		self.history = history;
		self.count_of_move_started += 1;
		self.moved = false;
		Ok(())
	}
	fn think<L,S>(&mut self,limit:&UsiGoTimeLimit,event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
			info_sender:S,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
		-> Result<BestMove,CommonError>
		where L: Logger, S: InfoSender, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		let kyokumen = self.kyokumen.as_ref().map(|k| k.clone()).ok_or(
			UsiProtocolError::InvalidState(
						String::from("Position information is not initialized."))
		)?;

		let teban = kyokumen.teban;
		let state = &kyokumen.state;
		let mc = &kyokumen.mc;

		let limit = limit.to_instant(teban);
		let (mhash,shash) = (self.mhash.clone(), self.shash.clone());
		let kyokumen_hash_map = self.kyokumen_hash_map.clone();

		let base_depth = self.base_depth;

		let mut info_sender = info_sender;

		let result = match self.alphabeta(&*event_queue,
					&mut info_sender, &on_error_handler,
					teban, state, Score::NEGINFINITE,
					Score::INFINITE, None, mc,
					None, &kyokumen_hash_map,
					&mut TwoKeyHashMap::new(),
					&mut TwoKeyHashMap::new(),mhash,shash,
					limit, base_depth, 0, base_depth) {
			Evaluation::Result(_,Some(m)) => {
				BestMove::Move(m,None)
			},
			Evaluation::Result(_,None) => {
				BestMove::Resign
			},
			Evaluation::Timeout(Some(m)) => {
				BestMove::Move(m,None)
			}
			Evaluation::Timeout(None) if self.quited => {
				BestMove::Abort
			},
			Evaluation::Timeout(None) => {
				BestMove::Resign
			},
			Evaluation::Error => {
				BestMove::Resign
			}
		};

		if let BestMove::Move(m,_) = result {
			let h = match self.history.last() {
				Some(&(ref banmen,ref mc,mhash,shash)) => {
					let (next,nmc,o) = Rule::apply_move_none_check(&State::new(banmen.clone()),teban,mc,m.to_applied_move());
					self.moved = true;
					let mut mhash = self.calc_main_hash(mhash,&teban,banmen,mc,&m,&o);
					let mut shash = self.calc_sub_hash(shash,&teban,banmen,mc,&m,&o);
					(next.get_banmen().clone(),nmc.clone(),mhash,shash)
				},
				None => {
					return Err(CommonError::Fail(String::from("The history of banmen has not been set yet.")));
				}
			};
			self.history.push(h);
		}

		Ok(result)
	}
	fn think_mate<L,S>(&mut self,_:&UsiGoMateTimeLimit,_:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
			_:S,_:Arc<Mutex<OnErrorHandler<L>>>)
		-> Result<CheckMate,CommonError>
		where L: Logger, S: InfoSender,
			Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {
		Ok(CheckMate::NotiImplemented)
	}
	fn on_stop(&mut self,_:&UserEvent) -> Result<(), CommonError> where CommonError: PlayerError {
		self.stop = true;
		Ok(())
	}
	fn gameover<L>(&mut self,s:&GameEndState,
		event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
		_:Arc<Mutex<OnErrorHandler<L>>>) -> Result<(),CommonError> where L: Logger, Arc<Mutex<OnErrorHandler<L>>>: Send + 'static {

		let kyokumen = self.kyokumen.as_ref().map(|k| k.clone()).ok_or(
			UsiProtocolError::InvalidState(
						String::from("Information of 'teban' is not set."))
		)?;

		let teban = kyokumen.teban;

		if self.count_of_move_started > 0 {
			let (teban,last_teban) = if self.moved {
				(teban,teban.opposite())
			} else {
				(teban,teban)
			};

			match self.evalutor {
				Some(ref mut evalutor) => {
					evalutor.learning(true,teban,last_teban,self.history.clone(),s,&*event_queue)?;
				},
				None => (),
			}
			self.history = Vec::new();
		}
		Ok(())
	}
	fn on_quit(&mut self,_:&UserEvent) -> Result<(), CommonError> where CommonError: PlayerError {
		self.quited = true;
		self.stop = true;
		Ok(())
	}

	fn quit(&mut self) -> Result<(),CommonError> {
		Ok(())
	}
}
