use std::collections::HashMap;
use std::fmt;
use rand;
use rand::Rng;
use std::sync::Arc;
use std::sync::Mutex;
use error::*;
use std::num::Wrapping;
use std::time::{Instant,Duration};
use std::cmp::Ordering;
use std::ops::Neg;

use usiagent::player::*;
use usiagent::command::*;
use usiagent::event::*;
use usiagent::shogi::*;
use usiagent::OnErrorHandler;
use usiagent::Logger;
use usiagent::error::PlayerError;
use usiagent::TryFrom;

use nn::Intelligence;

use hash::TwoKeyHashMap;

const KOMA_KIND_MAX:usize = KomaKind::Blank as usize;
const MOCHIGOMA_KIND_MAX:usize = MochigomaKind::Hisha as usize;
const MOCHIGOMA_MAX:usize = 18;
const SUJI_MAX:usize = 9;
const DAN_MAX:usize = 9;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
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
#[derive(Clone, Copy, Eq, PartialOrd, PartialEq, Debug)]
enum Score {
	NEGINFINITE,
	Value(i32),
	INFINITE,
}
impl Neg for Score {
	type Output = Score;

	fn neg(self) -> Score {
		match self {
			Score::INFINITE => Score::NEGINFINITE,
			Score::NEGINFINITE => Score::INFINITE,
			Score::Value(v) => Score::Value(-v),
		}
	}
}
const BASE_DEPTH:u32 = 1;
const MAX_DEPTH:u32 = 7;

pub struct NNShogiPlayer {
	stop:bool,
	pub quited:bool,
	kyokumen_hash_seeds:[[u64; SUJI_MAX * DAN_MAX]; KOMA_KIND_MAX + 1],
	mochigoma_hash_seeds:[[[u64; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX]; 2],
	pub teban:Option<Teban>,
	banmen:Option<Banmen>,
	mc:Option<MochigomaCollections>,
	mhash:u64,
	shash:u64,
	kyokumen_hash_map:TwoKeyHashMap<u32>,
	tinc:u32,
	evalutor:Intelligence,
	pub history:Vec<(Banmen,MochigomaCollections)>,
}
impl fmt::Debug for NNShogiPlayer {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "NNShogiPlayer")
	}
}
impl NNShogiPlayer {
	pub fn new() -> NNShogiPlayer {
		let mut rnd = rand::XorShiftRng::new_unseeded();

		let mut kyokumen_hash_seeds:[[u64; SUJI_MAX * DAN_MAX]; KOMA_KIND_MAX + 1] = [[0; SUJI_MAX * DAN_MAX]; KOMA_KIND_MAX + 1];
		let mut mochigoma_hash_seeds:[[[u64; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX]; 2] = [[[0; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX]; 2];

		for i in 0..(KOMA_KIND_MAX + 1) {
			for j in 0..(SUJI_MAX * DAN_MAX) {
				kyokumen_hash_seeds[i][j] = rnd.next_u64();
			}
		}

		for i in 0..MOCHIGOMA_MAX {
			for j in 0..(MOCHIGOMA_KIND_MAX + 1) {
				mochigoma_hash_seeds[0][i][j] = rnd.next_u64();
				mochigoma_hash_seeds[1][i][j] = rnd.next_u64();
			}
		}

		NNShogiPlayer {
			stop:false,
			quited:false,
			kyokumen_hash_seeds:kyokumen_hash_seeds,
			mochigoma_hash_seeds:mochigoma_hash_seeds,
			teban:None,
			banmen:None,
			mc:None,
			mhash:0,
			shash:0,
			kyokumen_hash_map:TwoKeyHashMap::new(),
			tinc:0,
			evalutor:Intelligence::new(String::from("data")),
			history:Vec::new(),
		}
	}

	fn send_message<L>(&mut self, info_sender:&USIInfoSender,
			on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>, message:&str)
		where L: Logger {
		let mut commands:Vec<UsiInfoSubCommand> = Vec::new();
		commands.push(UsiInfoSubCommand::Str(String::from(message)));

		match info_sender.send(commands) {
			Ok(_) => (),
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}
	}

	fn send_seldepth<L>(&mut self, info_sender:&USIInfoSender,
			on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>, depth:u32, seldepth:u32)
		where L: Logger {
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

	fn alphabeta<'a,L>(&mut self,
			event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>,
								info_sender:&USIInfoSender,
								on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
								teban:Teban,banmen:&Banmen,
								mut alpha:Score,beta:Score,
								m:Option<Move>,mc:&MochigomaCollections,
								obtained:Option<ObtainKind>,
								current_kyokumen_hash_map:&TwoKeyHashMap<u32>,
								already_oute_hash_map:&mut TwoKeyHashMap<()>,
								ignore_oute_hash_map:&mut TwoKeyHashMap<()>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,depth:u32,current_depth:u32) -> Evaluation where L: Logger {
		self.send_seldepth(info_sender, on_error_handler, BASE_DEPTH, current_depth);

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

		if depth == 0 || current_depth == MAX_DEPTH {
			if (limit.is_some() &&
				limit.unwrap() - Instant::now() <= Duration::from_millis(10)) || self.stop {
				self.send_message(info_sender, on_error_handler, "think timeout!");
				return Evaluation::Timeout(None);
			} else {
				let s = match self.evalutor.evalute(teban,banmen,mc) {
					Ok(s) => s,
					Err(ref e) => {
						on_error_handler.lock().map(|h| h.call(e)).is_err();
						return Evaluation::Error;
					}
				};
				self.send_message(info_sender, on_error_handler, &format!("evalute sore = {}",s));
				return Evaluation::Result(Score::Value(s),m);
			}
		}

		let mvs:Vec<LegalMove> = banmen.legal_moves_all(&teban, mc);

		match self.handle_events(event_queue, on_error_handler) {
			Ok(_) => (),
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}

		if mvs.len() == 0 {
			return Evaluation::Result(Score::NEGINFINITE,None);
		}

		if self.stop {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return Evaluation::Timeout(Some(mvs[0].to_move()));
		}

		let mut mvs:Vec<(bool,LegalMove)> = mvs.into_iter().map(|m| {
			match m {
				LegalMove::To(_,_,Some(ObtainKind::Ou)) => {
					(true,m)
				},
				LegalMove::To(ref s,ref d,_) => {
					match banmen.apply_move_none_check(&teban,mc,&Move::To(*s,*d)) {
						(ref b,_,_) => (b.win_only_moves(&teban).len() > 0,m)
					}
				},
				LegalMove::Put(ref k,ref d) => {
					match banmen.apply_move_none_check(&teban,mc,&Move::Put(*k,*d)) {
						(ref b,_,_) => (b.win_only_moves(&teban).len() > 0,m)
					}
				}
			}
		}).collect::<Vec<(bool,LegalMove)>>();

		match self.handle_events(event_queue, on_error_handler) {
			Ok(_) => (),
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}

		if (limit.is_some() &&
			limit.unwrap() - Instant::now() <= Duration::from_millis(10)) || self.stop {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return Evaluation::Timeout(Some(mvs[0].1.to_move()));
		}

		mvs.sort_by(|a,b| {
			match a {
				&(_,LegalMove::To(_,_,Some(ObtainKind::Ou))) => {
					return Ordering::Less;
				},
				_ => {
					match b {
						&(_,LegalMove::To(_,_,Some(ObtainKind::Ou))) => {
							return Ordering::Greater;
						},
						_ => (),
					}
				}
			}

			let a_is_oute = match a {
				&(f,_) => f
			};

			let b_is_oute = match b {
				&(f,_) => f
			};

			if a_is_oute == b_is_oute {
				Ordering::Equal
			} else if a_is_oute {
				Ordering::Less
			} else {
				Ordering::Greater
			}
		});

		match mvs[0] {
			(_,LegalMove::To(ref s,ref d,Some(ObtainKind::Ou))) => {
				return Evaluation::Result(Score::INFINITE,Some(Move::To(*s,*d)));
			},
			_ => (),
		}

		for m in &mvs {
			match self.handle_events(event_queue, on_error_handler) {
				Ok(_) => (),
				Err(ref e) => {
					on_error_handler.lock().map(|h| h.call(e)).is_err();
				}
			}

			if (limit.is_some() &&
				limit.unwrap() - Instant::now() <= Duration::from_millis(10)) || self.stop {
				self.send_message(info_sender, on_error_handler, "think timeout!");
				return Evaluation::Timeout(Some(m.1.to_move()));
			}
			match *m {
				(true,m) => {
					let (mhash,shash) = match m {
						LegalMove::To(ref s, ref d,ref o) => {
							let o = match o {
								&Some(o) => {
									match MochigomaKind::try_from(o) {
										Ok(o) => Some(o),
										Err(_) => None,
									}
								},
								&None => None,
							};

							let m = Move::To(*s,*d);

							(
								self.calc_main_hash(mhash,&teban,banmen,mc,&m,&o),
								self.calc_sub_hash(shash,&teban,banmen,mc,&m,&o)
							)
						},
						LegalMove::Put(ref k, ref d) => {
							let m = Move::Put(*k,*d);

							(
								self.calc_main_hash(mhash,&teban,banmen,mc,&m,&None),
								self.calc_sub_hash(shash,&teban,banmen,mc,&m,&None)
							)
						}
					};

					let mut current_kyokumen_hash_map = current_kyokumen_hash_map.clone();

					match current_kyokumen_hash_map.get(&mhash,&shash) {
						Some(c) if c >= 3 => {
							continue;
						},
						Some(c) => {
							current_kyokumen_hash_map.insert(mhash,shash,c+1);
						},
						None => (),
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
					let next = match m {
						LegalMove::To(ref s, ref d,_) => {
							banmen.apply_move_none_check(&teban,mc,&Move::To(*s,*d))
						},
						LegalMove::Put(ref k, ref d) => {
							banmen.apply_move_none_check(&teban,mc,&Move::Put(*k,*d))
						}
					};

					match next {
						(ref next,ref mc,_) if next.win_only_moves(&teban.opposite()).len() == 0 => {
							let is_put_fu = match m {
								LegalMove::Put(MochigomaKind::Fu,_) => true,
								_ => false,
							};

							match self.handle_events(event_queue, on_error_handler) {
								Ok(_) => (),
								Err(ref e) => {
									on_error_handler.lock().map(|h| h.call(e)).is_err();
								}
							}

							if (limit.is_some() &&
								limit.unwrap() - Instant::now() <= Duration::from_millis(10)) || self.stop {
								self.send_message(info_sender, on_error_handler, "think timeout!");
								return Evaluation::Timeout(Some(m.to_move()));
							}

							match self.respond_oute_only(event_queue,
																info_sender,
																on_error_handler,
																teban.opposite(),next,mc,
																is_put_fu,&current_kyokumen_hash_map,
																already_oute_hash_map,
																ignore_oute_hash_map,
																mhash,shash,limit,current_depth+1) {
								OuteEvaluation::Result(d) if d >= 0 &&
																is_put_fu &&
																	d - current_depth as i32 == 2 => {
									ignore_oute_hash_map.insert(mhash,shash,());
									return Evaluation::Result(Score::INFINITE,Some(m.to_move()));
								},
								OuteEvaluation::Result(d) if d >= 0 => {
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
				},
				(false,_) => {
					break;
				}
			}
		}

		let mut scoreval = Score::NEGINFINITE;
		let mut best_move:Option<Move> = None;

		for m in &mvs {
			match self.handle_events(event_queue, on_error_handler) {
				Ok(_) => (),
				Err(ref e) => {
					on_error_handler.lock().map(|h| h.call(e)).is_err();
				}
			}

			if (limit.is_some() &&
				limit.unwrap() - Instant::now() <= Duration::from_millis(10)) || self.stop {
				self.send_message(info_sender, on_error_handler, "think timeout!");
				return match best_move {
					Some(best_move) => Evaluation::Timeout(Some(best_move)),
					None => Evaluation::Timeout(Some(m.1.to_move())),
				};
			}
			match *m {
				(_,m) => {
					let (next,obtained) = match m {
						LegalMove::To(ref s, ref d,ref o) => {
							(banmen.apply_move_none_check(&teban,mc,&Move::To(*s,*d)),*o)
						},
						LegalMove::Put(ref k, ref d) => {
							(banmen.apply_move_none_check(&teban,mc,&Move::Put(*k,*d)),None)
						}
					};

					let depth = match obtained {
						Some(_) => depth + 1,
						None => depth,
					};

					match next {
						(ref banmen,ref mc,_) if banmen.win_only_moves(&teban.opposite()).len() == 0 => {
							match self.alphabeta(event_queue,
								info_sender,
								on_error_handler,
								teban.opposite(),&banmen,
								-beta,-alpha,Some(m.to_move()),&mc,
								obtained,&current_kyokumen_hash_map,
								already_oute_hash_map,
								ignore_oute_hash_map,
								mhash,shash,limit,depth-1,current_depth+1) {

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
						},
						_ => (),
					}
				}
			}
		}

		Evaluation::Result(scoreval,best_move)
	}

	fn respond_oute_only<'a,L>(&mut self,
								event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>,
								info_sender:&USIInfoSender,
								on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
								teban:Teban,banmen:&Banmen,
								mc:&MochigomaCollections,is_put_fu:bool,
								current_kyokumen_hash_map:&TwoKeyHashMap<u32>,
								already_oute_hash_map:&mut TwoKeyHashMap<()>,
								ignore_oute_hash_map:&mut TwoKeyHashMap<()>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,current_depth:u32) -> OuteEvaluation where L: Logger {
		let mvs = banmen.respond_oute_only_moves_all(&teban, mc);

		self.send_seldepth(info_sender, on_error_handler, BASE_DEPTH, current_depth);

		match self.handle_events(event_queue, on_error_handler) {
			Ok(_) => (),
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}

		if (limit.is_some() &&
			limit.unwrap() - Instant::now() <= Duration::from_millis(10)) || self.stop {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return OuteEvaluation::Timeout;
		}

		if mvs.len() == 0 {
			return OuteEvaluation::Result(current_depth as i32)
		} else {
			for m in &mvs {
				match self.handle_events(event_queue, on_error_handler) {
					Ok(_) => (),
					Err(ref e) => {
						on_error_handler.lock().map(|h| h.call(e)).is_err();
					}
				}
				if (limit.is_some() &&
					limit.unwrap() - Instant::now() <= Duration::from_millis(10)) || self.stop {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return OuteEvaluation::Timeout;
				}

				let (mhash,shash) = match m {
					&LegalMove::To(ref s, ref d,ref o) => {
						let o = match o {
							&Some(o) => {
								match MochigomaKind::try_from(o) {
									Ok(o) => Some(o),
									Err(_) => None,
								}
							},
							&None => None,
						};

						let m = Move::To(*s,*d);

						(
							self.calc_main_hash(mhash,&teban,banmen,mc,&m,&o),
							self.calc_sub_hash(shash,&teban,banmen,mc,&m,&o)
						)
					},
					&LegalMove::Put(ref k, ref d) => {
						let m = Move::Put(*k,*d);

						(
							self.calc_main_hash(mhash,&teban,banmen,mc,&m,&None),
							self.calc_sub_hash(shash,&teban,banmen,mc,&m,&None)
						)
					}
				};

				match already_oute_hash_map.get(&mhash,&shash) {
					None => {
						already_oute_hash_map.insert(mhash,shash,());
					},
					Some(_) => {
						return OuteEvaluation::Result(-1);
					}
				}

				let next = match m {
					&LegalMove::To(ref s, ref d,_) => {
						banmen.apply_move_none_check(&teban,mc,&Move::To(*s,*d))
					},
					&LegalMove::Put(ref k, ref d) => {
						banmen.apply_move_none_check(&teban,mc,&Move::Put(*k,*d))
					}
				};

				match next {
					(ref next,ref mc,_) => {
						match self.oute_only(event_queue,
												info_sender,
												on_error_handler,
												teban.opposite(),next,mc,
												is_put_fu,current_kyokumen_hash_map,
												already_oute_hash_map,
												ignore_oute_hash_map,
												mhash,shash,limit,current_depth+1) {
							OuteEvaluation::Result(d) if d >= 0 && !is_put_fu => {
								return OuteEvaluation::Result(d);
							},
							OuteEvaluation::Result(d) => {
								return OuteEvaluation::Result(d);
							},
							OuteEvaluation::Timeout => {
								return OuteEvaluation::Timeout;
							},
						}
					}
				}
			}

			OuteEvaluation::Result(-1)
		}
	}

	fn oute_only<'a,L>(&mut self,
								event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>,
								info_sender:&USIInfoSender,
								on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
								teban:Teban,banmen:&Banmen,
								mc:&MochigomaCollections,is_put_fu:bool,
								current_kyokumen_hash_map:&TwoKeyHashMap<u32>,
								already_oute_hash_map:&mut TwoKeyHashMap<()>,
								ignore_oute_hash_map:&mut TwoKeyHashMap<()>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,current_depth:u32) -> OuteEvaluation where L: Logger {
		let mvs = banmen.oute_only_moves_all(&teban, mc);

		self.send_seldepth(info_sender, on_error_handler, BASE_DEPTH, current_depth);

		match self.handle_events(event_queue, on_error_handler) {
			Ok(_) => (),
			Err(ref e) => {
				on_error_handler.lock().map(|h| h.call(e)).is_err();
			}
		}
		if (limit.is_some() &&
			limit.unwrap() - Instant::now() <= Duration::from_millis(10)) || self.stop {
			self.send_message(info_sender, on_error_handler, "think timeout!");
			return OuteEvaluation::Timeout;
		}

		if mvs.len() == 0 {
			OuteEvaluation::Result(-1)
		} else {
			for m in &mvs {
				match self.handle_events(event_queue, on_error_handler) {
					Ok(_) => (),
					Err(ref e) => {
						on_error_handler.lock().map(|h| h.call(e)).is_err();
					}
				}
				if (limit.is_some() &&
					limit.unwrap() - Instant::now() <= Duration::from_millis(10)) || self.stop {
					self.send_message(info_sender, on_error_handler, "think timeout!");
					return OuteEvaluation::Timeout;
				}

				let next = match m {
					&LegalMove::To(ref s, ref d,_) => {
						banmen.apply_move_none_check(&teban,mc,&Move::To(*s,*d))
					},
					&LegalMove::Put(ref k, ref d) => {
						banmen.apply_move_none_check(&teban,mc,&Move::Put(*k,*d))
					}
				};

				let (mhash,shash) = match m {
					&LegalMove::To(ref s, ref d,ref o) => {
						let o = match o {
							&Some(o) => {
								match MochigomaKind::try_from(o) {
									Ok(o) => Some(o),
									Err(_) => None,
								}
							},
							&None => None,
						};

						let m = Move::To(*s,*d);

						(
							self.calc_main_hash(mhash,&teban,banmen,mc,&m,&o),
							self.calc_sub_hash(shash,&teban,banmen,mc,&m,&o)
						)
					},
					&LegalMove::Put(ref k, ref d) => {
						let m = Move::Put(*k,*d);

						(
							self.calc_main_hash(mhash,&teban,banmen,mc,&m,&None),
							self.calc_sub_hash(shash,&teban,banmen,mc,&m,&None)
						)
					}
				};

				match already_oute_hash_map.get(&mhash,&shash) {
					None => {
						already_oute_hash_map.insert(mhash,shash,());
					},
					Some(_) => {
						return OuteEvaluation::Result(-1);
					}
				}

				match next {
					(ref next,ref mc,_) => {
						match self.respond_oute_only(event_queue,
														info_sender,
														on_error_handler,
														teban.opposite(),next,mc,
														is_put_fu,current_kyokumen_hash_map,
														already_oute_hash_map,
														ignore_oute_hash_map,
														mhash,shash,limit,current_depth+1) {
							OuteEvaluation::Result(d) if d >= 0 && !is_put_fu => {
								return OuteEvaluation::Result(d);
							},
							OuteEvaluation::Result(d) => {
								return OuteEvaluation::Result(d);
							},
							OuteEvaluation::Timeout => {
								return OuteEvaluation::Timeout;
							}
						}
					}
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
					&Move::To(KomaSrcPosition(sx,sy), KomaDstToPosition(dx, dy, _)) => {
						let sx = (9 - sx) as usize;
						let sy = (sy - 1) as usize;
						let dx = (9 - dx) as usize;
						let dy = dy as usize - 1;

						let mut hash = h;
						let k = kinds[sy][sx] as usize;

						hash =  pull(hash,self.kyokumen_hash_seeds[k][sy * 8 + sx]);
						hash = add(hash,self.kyokumen_hash_seeds[KomaKind::Blank as usize][sy * 8 + sx]);

						let dk = kinds[dy][dx] as usize;

						hash =  pull(hash,self.kyokumen_hash_seeds[dk][dy * 8 + dx]);
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
								hash = pull(hash,self.mochigoma_hash_seeds[0][c][k]);
							},
							&Teban::Gote => {
								hash = pull(hash,self.mochigoma_hash_seeds[1][c][k]);
							}
						}

						let dx = md.0 as usize;
						let dy = md.1 as usize - 1;

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

		Ok(kinds)
	}
	fn get_options(&mut self) -> Result<HashMap<String,UsiOptType>,CommonError> {
		Ok(HashMap::new())
	}
	fn take_ready(&mut self) -> Result<(),CommonError> {
		Ok(())
	}
	fn set_option(&mut self,_:String,_:SysEventOption) -> Result<(),CommonError> {
		Ok(())
	}
	fn newgame(&mut self) -> Result<(),CommonError> {
		self.teban = None;
		self.banmen = None;
		self.mc = None;
		self.history.clear();
		if !self.quited {
			self.stop = false;
		}
		Ok(())
	}
	fn set_position(&mut self,teban:Teban,ban:Banmen,
					ms:HashMap<MochigomaKind,u32>,mg:HashMap<MochigomaKind,u32>,_:u32,m:Vec<Move>)
		-> Result<(),CommonError> {
		self.history.clear();
		self.kyokumen_hash_map.clear();

		let (mut mhash,mut shash) = self.calc_initial_hash(&ban,&ms,&mg);

		self.kyokumen_hash_map.insert(mhash,shash,1);

		let mut banmen = ban.clone();
		let mut t = teban.clone();

		let mut mc = if ms.len() == 0 && mg.len() == 0 {
			MochigomaCollections::Empty
		} else {
			MochigomaCollections::Pair(ms,mg)
		};
		self.history.push((banmen.clone(),mc.clone()));

		for m in &m {
			 match banmen.apply_move_none_check(&t,&mc,&m) {
				(next,nmc,o) => {
					mhash = self.calc_main_hash(mhash,&t,&next,&nmc,m,&o);
					shash = self.calc_sub_hash(shash,&t,&next,&nmc,m,&o);

					match self.kyokumen_hash_map.get(&mhash,&shash) {
						Some(c) => {
							self.kyokumen_hash_map.insert(mhash,shash,c+1);
						},
						None => {
							self.kyokumen_hash_map.insert(mhash,shash,1);
						}
					}
					self.history.push((banmen,mc));
					banmen = next;
					mc = nmc;
					t = t.opposite();
				}
			}
		}
		self.teban = Some(t);
		self.banmen = Some(banmen);
		self.mc = Some(mc);
		self.mhash = mhash;
		self.shash = shash;
		Ok(())
	}
	fn think<L>(&mut self,limit:&UsiGoTimeLimit,event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
			info_sender:&USIInfoSender,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
			-> Result<BestMove,CommonError> where L: Logger {
		match self.teban {
			Some(teban) => {
				let now = Instant::now();
				let limit = match limit {
					&UsiGoTimeLimit::None => None,
					&UsiGoTimeLimit::Infinite => None,
					&UsiGoTimeLimit::Limit(Some((ms,mg)),None) => {
						Some(match teban {
							Teban::Sente => {
								now + Duration::from_millis(ms as u64)
							},
							Teban::Gote => {
								now + Duration::from_millis(mg as u64)
							}
						})
					},
					&UsiGoTimeLimit::Limit(Some((ms,mg)),Some(UsiGoByoyomiOrInc::Byoyomi(b))) => {
						Some(match teban {
							Teban::Sente => {
								now + Duration::from_millis(ms as u64 + b as u64)
							},
							Teban::Gote => {
								now + Duration::from_millis(mg as u64 + b as u64)
							}
						})
					}
					&UsiGoTimeLimit::Limit(Some((ms,mg)),Some(UsiGoByoyomiOrInc::Inc(bs,bg))) => {
						Some(match teban {
							Teban::Sente => {
								let tinc = self.tinc + bs as u32;
								self.tinc = tinc;
								now + Duration::from_millis(ms as u64 + tinc as u64)
							},
							Teban::Gote => {
								let tinc = self.tinc + bg as u32;
								self.tinc = tinc;
								now + Duration::from_millis(mg as u64 + tinc as u64)
							}
						})
					},
					&UsiGoTimeLimit::Limit(None,Some(UsiGoByoyomiOrInc::Byoyomi(b))) => {
						Some(now + Duration::from_millis(b as u64))
					}
					&UsiGoTimeLimit::Limit(None,Some(UsiGoByoyomiOrInc::Inc(bs,bg))) => {
						Some(match teban {
							Teban::Sente => {
								now + Duration::from_millis(bs as u64)
							},
							Teban::Gote => {
								now + Duration::from_millis(bg as u64)
							}
						})
					},
					&UsiGoTimeLimit::Limit(None,None) => {
						Some(now)
					}
				};
				let banmen = match self.banmen {
					Some(Banmen(kinds)) => Some(Banmen(kinds.clone())),
					None => None,
				};
				match banmen {
					Some(ref banmen) => {
						let mc = match self.mc {
							Some(ref mc) => Some(mc.clone()),
							None => None,
						};
						match mc {
							Some(ref mc) => {
								let (mhash,shash) = (self.mhash.clone(), self.shash.clone());
								let kyokumen_hash_map = self.kyokumen_hash_map.clone();

								let result = match self.alphabeta(&*event_queue,
											info_sender, &on_error_handler,
											teban, &banmen, Score::NEGINFINITE,
											Score::INFINITE, None, &mc,
											None, &kyokumen_hash_map,
											&mut TwoKeyHashMap::new(),
											&mut TwoKeyHashMap::new(),mhash,shash,
											limit, BASE_DEPTH, 0) {
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
								match limit {
									Some(limit) => {
										self.tinc += (limit - Instant::now()).subsec_nanos() * 1000000;
									},
									None => (),
								}
								return Ok(result);
							},
							None => (),
						}
					},
					None => (),
				};
			},
			None => (),
		}

		Err(CommonError::Fail(String::from("Initialization of position info has not been completed.")))
	}
	fn think_mate<L>(&mut self,_:&UsiGoMateTimeLimit,_:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
			_:&USIInfoSender,_:Arc<Mutex<OnErrorHandler<L>>>)
			-> Result<CheckMate,CommonError> where L: Logger {
		Ok(CheckMate::NotiImplemented)
	}
	fn on_stop(&mut self,_:&UserEvent) -> Result<(), CommonError> where CommonError: PlayerError {
		self.stop = true;
		Ok(())
	}
	fn gameover<L>(&mut self,s:&GameEndState,
		event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
		_:&Arc<Mutex<OnErrorHandler<L>>>) -> Result<(),CommonError> where L: Logger {

		let teban = match self.teban {
			Some(teban) =>  teban,
			None => {
				return Err(CommonError::Fail(String::from("Information of 'teban' is not set.")));
			}
		};

		self.evalutor.learning(teban,self.history.clone(),s,&*event_queue)?;
		self.history = Vec::new();

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
