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
}
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum OuteEvaluation {
	Result(i32),
	Timeout,
}
#[derive(Clone, Copy, Eq, PartialOrd, PartialEq, Debug)]
enum Score {
	INFINITE,
	Value(i32),
	NEGINFINITE,
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
pub struct NNShogiPlayer {
	stop:bool,
	quited:bool,
	kyokumen_hash_seeds:[[u64; KOMA_KIND_MAX + 1]; SUJI_MAX * DAN_MAX],
	mochigoma_hash_seeds:[[[u64; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX]; 2],
	teban:Option<Teban>,
	banmen:Option<Banmen>,
	mc:Option<MochigomaCollections>,
	mhash:u64,
	shash:u64,
	kyokumen_hash_map:TwoKeyHashMap<u32>,
	tinc:u32,
	evalutor:Intelligence,
}
impl fmt::Debug for NNShogiPlayer {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "NNShogiPlayer")
	}
}
impl NNShogiPlayer {
	pub fn new() -> NNShogiPlayer {
		let mut rnd = rand::XorShiftRng::new_unseeded();

		let mut kyokumen_hash_seeds:[[u64; KOMA_KIND_MAX + 1]; SUJI_MAX * DAN_MAX] = [[0; KOMA_KIND_MAX + 1]; SUJI_MAX * DAN_MAX];
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

	fn alphabeta<'a,L>(&mut self,
			event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>,
								info_sender:&USIInfoSender,
								on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
								teban:Teban,banmen:&Banmen,
								mut alpha:Score,beta:Score,
								m:Option<Move>,mc:&MochigomaCollections,
								obtained:Option<ObtainKind>,
								current_kyokumen_hash_map:&TwoKeyHashMap<u32>,
								already_oute_hash_map:&mut TwoKeyHashMap<u32>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,depth:u32,current_depth:u32) -> Evaluation where L: Logger {
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

		if depth == 0 {
			if (limit.is_some() &&
				limit.unwrap() - Instant::now() <= Duration::from_millis(10)) || self.stop {
				self.send_message(info_sender, on_error_handler, "think timeout!");
				return Evaluation::Timeout(None);
			} else {
				return Evaluation::Result(Score::Value(self.evalutor.evalute(teban,banmen,mc)),m);
			}
		}

		let mvs:Vec<LegalMove> = banmen.oute_only_moves_all(&teban, mc);

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
							already_oute_hash_map.insert(mhash,shash,current_depth);
						},
						Some(depth) if depth - current_depth == 2 => {
							match m {
								LegalMove::Put(MochigomaKind::Fu,_) => {
									continue;
								},
								_ => (),
							}
						}
						Some(_) => (),
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
						(ref next,ref mc,_) => {
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
																mhash,shash,limit,current_depth+1) {
								OuteEvaluation::Result(d) if d >= 0 &&
																is_put_fu &&
																	d - current_depth as i32 != 2 => {
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
						}
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

					let current_depth = match obtained {
						Some(_) => current_depth + 1,
						None => current_depth,
					};

					match next {
						(banmen,mc,_) => {
							match self.alphabeta(event_queue,
								info_sender,
								on_error_handler,
								teban.opposite(),&banmen,
								-beta,-alpha,Some(m.to_move()),&mc,
								obtained,&current_kyokumen_hash_map,
								already_oute_hash_map,mhash,shash,limit,depth-1,current_depth-1) {

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
								}
							}
						}
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
								already_oute_hash_map:&mut TwoKeyHashMap<u32>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,current_depth:u32) -> OuteEvaluation where L: Logger {
		let mvs = banmen.respond_oute_only_moves_all(&teban, mc);

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
			let mut longest_depth = -1;

			for m in mvs {
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

				match already_oute_hash_map.get(&mhash,&shash) {
					None => {
						already_oute_hash_map.insert(mhash,shash,current_depth);
					},
					Some(_) => {
						return OuteEvaluation::Result(-1);
					}
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
					(ref next,ref mc,_) => {
						match self.oute_only(event_queue,
												info_sender,
												on_error_handler,
												teban.opposite(),next,mc,
												is_put_fu,current_kyokumen_hash_map,
												already_oute_hash_map,mhash,shash,limit,current_depth+1) {
							OuteEvaluation::Result(d) if d >= 0 && !is_put_fu => {
								return OuteEvaluation::Result(d);
							},
							OuteEvaluation::Result(d) if d >= 0 => {
								longest_depth = longest_depth.max(d);
							},
							OuteEvaluation::Result(_) => {
								return OuteEvaluation::Result(-1);
							},
							OuteEvaluation::Timeout => {
								return OuteEvaluation::Timeout;
							},
						}
					}
				}
			}

			OuteEvaluation::Result(longest_depth)
		}
	}

	fn oute_only<'a,L>(&mut self,
								event_queue:&'a Mutex<EventQueue<UserEvent,UserEventKind>>,
								info_sender:&USIInfoSender,
								on_error_handler:&Arc<Mutex<OnErrorHandler<L>>>,
								teban:Teban,banmen:&Banmen,
								mc:&MochigomaCollections,is_put_fu:bool,
								current_kyokumen_hash_map:&TwoKeyHashMap<u32>,
								already_oute_hash_map:&mut TwoKeyHashMap<u32>,
								mhash:u64,shash:u64,
								limit:Option<Instant>,current_depth:u32) -> OuteEvaluation where L: Logger {
		let mvs = banmen.oute_only_moves_all(&teban, mc);

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
			let mut shortest_depth = -1;

			for m in mvs {
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
					LegalMove::To(ref s, ref d,_) => {
						banmen.apply_move_none_check(&teban,mc,&Move::To(*s,*d))
					},
					LegalMove::Put(ref k, ref d) => {
						banmen.apply_move_none_check(&teban,mc,&Move::Put(*k,*d))
					}
				};

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

				match already_oute_hash_map.get(&mhash,&shash) {
					None => {
						already_oute_hash_map.insert(mhash,shash,current_depth);
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
														mhash,shash,limit,current_depth+1) {
							OuteEvaluation::Result(d) if d >= 0 && !is_put_fu => {
								return OuteEvaluation::Result(d);
							},
							OuteEvaluation::Result(d) if d >= 0 => {
								if shortest_depth == -1 {
									shortest_depth = d;
								} else {
									shortest_depth = shortest_depth.min(d);
								}
							},
							OuteEvaluation::Timeout => {
								return OuteEvaluation::Timeout;
							},
							_ => (),
						}
					}
				}
			}
			OuteEvaluation::Result(shortest_depth)
		}
	}

	fn calc_hash<AF,PF>(&self,h:u64,t:&Teban,b:&Banmen,mc:&MochigomaCollections,
												m:&Move,obtained:&Option<MochigomaKind>,add:AF,pull:PF)
		-> u64 where AF: Fn(u64,u64) -> u64, PF: Fn(u64,u64) -> u64 {
		match b {
			&Banmen(ref kinds) => {
				match m {
					&Move::To(ref ms, KomaDstToPosition(dx, dy, _)) => {
						let sx = 9 - ms.0 as usize;
						let sy = ms.1 as usize;
						let dx = 9 - dx as usize;
						let dy = dy as usize;

						let mut hash = h;
						let k = kinds[sy][sx] as usize;

						hash =  pull(hash,self.kyokumen_hash_seeds[k][sx * 9 + sy]);
						hash = add(hash,self.kyokumen_hash_seeds[KomaKind::Blank as usize][sx * 9 + sy]);

						let dk = kinds[dy][dx] as usize;

						hash =  pull(hash,self.kyokumen_hash_seeds[dk][dx * 9 + dy]);
						hash = add(hash,self.kyokumen_hash_seeds[k][dx * 9 + dy]);

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
											Some(c) => (*c - 1) as usize,
											None => {
												return hash;
											}
										}
									},
									&MochigomaCollections::Empty => 0,
								}
							},
							&Teban::Gote => {
								match mc {
									&MochigomaCollections::Pair(_,ref mc) => {
										match mc.get(&mk) {
											Some(c) => (*c - 1) as usize,
											None => {
												return hash;
											}
										}
									},
									&MochigomaCollections::Empty => 0,
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
						let dy = md.1 as usize;

						hash = add(hash,self.kyokumen_hash_seeds[k + 1usize][dx * 9 + dy]);
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
}
impl USIPlayer<CommonError> for NNShogiPlayer {
	const ID: &'static str = "nnshogi";
	const AUTHOR: &'static str = "jinpu";
	fn get_option_kinds(&mut self) -> Result<HashMap<String,SysEventOptionKind>,CommonError> {
		Ok(HashMap::new())
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
		Ok(())
	}
	fn set_position(&mut self,teban:Teban,ban:Banmen,
					ms:HashMap<MochigomaKind,u32>,mg:HashMap<MochigomaKind,u32>,n:u32,m:Vec<Move>)
		-> Result<(),CommonError> {
		Ok(())
	}
	fn think<L>(&mut self,limit:&UsiGoTimeLimit,event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
			info_sender:&USIInfoSender,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
			-> Result<BestMove,CommonError> where L: Logger {
		match self.teban {
			Some(teban) => {
				let limit = match limit {
					&UsiGoTimeLimit::None => None,
					&UsiGoTimeLimit::Infinite => None,
					&UsiGoTimeLimit::Limit(Some((ms,mg)),None) => {
						Some(match teban {
							Teban::Sente => {
								Instant::now() + Duration::from_millis(ms as u64)
							},
							Teban::Gote => {
								Instant::now() + Duration::from_millis(mg as u64)
							}
						})
					},
					&UsiGoTimeLimit::Limit(Some((ms,mg)),Some(UsiGoByoyomiOrInc::Byoyomi(b))) => {
						Some(match teban {
							Teban::Sente => {
								Instant::now() + Duration::from_millis(ms as u64 + b as u64)
							},
							Teban::Gote => {
								Instant::now() + Duration::from_millis(mg as u64 + b as u64)
							}
						})
					}
					&UsiGoTimeLimit::Limit(Some((ms,mg)),Some(UsiGoByoyomiOrInc::Inc(bs,bg))) => {
						Some(match teban {
							Teban::Sente => {
								Instant::now() + Duration::from_millis(ms as u64 + bs as u64)
							},
							Teban::Gote => {
								Instant::now() + Duration::from_millis(mg as u64 + bg as u64)
							}
						})
					},
					&UsiGoTimeLimit::Limit(None,Some(UsiGoByoyomiOrInc::Byoyomi(b))) => {
						Some(Instant::now() + Duration::from_millis(b as u64))
					}
					&UsiGoTimeLimit::Limit(None,Some(UsiGoByoyomiOrInc::Inc(bs,bg))) => {
						Some(match teban {
							Teban::Sente => {
								Instant::now() + Duration::from_millis(bs as u64)
							},
							Teban::Gote => {
								Instant::now() + Duration::from_millis(bg as u64)
							}
						})
					},
					&UsiGoTimeLimit::Limit(None,None) => {
						Some(Instant::now())
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

								return Ok(match self.alphabeta(&*event_queue,
											info_sender, &on_error_handler,
											teban, &banmen, Score::NEGINFINITE,
											Score::INFINITE, None, &mc,
											None, &kyokumen_hash_map,
											&mut TwoKeyHashMap::new(),mhash,shash,
											limit, 1, 0) {
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
									}
								});
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
			info_sender:&USIInfoSender,on_error_handler:Arc<Mutex<OnErrorHandler<L>>>)
			-> Result<CheckMate,CommonError> where L: Logger {
		Ok(CheckMate::NotiImplemented)
	}
	fn on_stop(&mut self,e:&UserEvent) -> Result<(), CommonError> where CommonError: PlayerError {
		self.stop = true;
		Ok(())
	}
	fn gameover(&mut self,s:&GameEndState,event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>) -> Result<(),CommonError> {
		Ok(())
	}
	fn on_quit(&mut self,e:&UserEvent) -> Result<(), CommonError> where CommonError: PlayerError {
		self.quited = true;
		Ok(())
	}

	fn quit(&mut self) -> Result<(),CommonError> {
		Ok(())
	}
}
