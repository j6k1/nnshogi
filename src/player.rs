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
	Timeout,
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
	kyokumen_hash_seeds:[[u64; KOMA_KIND_MAX + 1]; SUJI_MAX * DAN_MAX],
	mochigoma_hash_seeds:[[u64; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX],
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
		let mut mochigoma_hash_seeds:[[u64; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX] = [[0; MOCHIGOMA_KIND_MAX + 1]; MOCHIGOMA_MAX];

		for i in 0..(KOMA_KIND_MAX + 1) {
			for j in 0..(SUJI_MAX * DAN_MAX) {
				kyokumen_hash_seeds[i][j] = rnd.next_u64();
			}
		}

		for i in 0..MOCHIGOMA_MAX {
			for j in 0..(MOCHIGOMA_KIND_MAX + 1) {
				mochigoma_hash_seeds[i][j] = rnd.next_u64();
			}
		}

		NNShogiPlayer {
			stop:false,
			kyokumen_hash_seeds:kyokumen_hash_seeds,
			mochigoma_hash_seeds:mochigoma_hash_seeds,
			evalutor:Intelligence::new(String::from("data")),
		}
	}

	fn alphabeta(&mut self, teban:Teban,banmen:&Banmen,
								mut alpha:Score,mut beta:Score,
								m:Move,mc:&MochigomaCollections,
								obtained:Option<ObtainKind>,
								current_kyokumen_hash_map:&TwoKeyHashMap<u32>,
								already_oute_hash_map:&mut TwoKeyHashMap<()>,
								mhash:u64,shash:u64,
								limit:Instant,depth:u32,current_depth:u32) -> Evaluation {
		match obtained {
			Some(ObtainKind::Ou) => {
				return Evaluation::Result(Score::NEGINFINITE,None);
			},
			_ => (),
		}

		if depth == 0 {
			if limit - Instant::now() <= Duration::from_millis(10) {
				return Evaluation::Timeout;
			} else {
				return Evaluation::Result(Score::Value(self.evalutor.evalute(teban,banmen,mc)),Some(m));
			}
		}

		let mut mvs:Vec<LegalMove> = banmen.oute_only_moves_all(&teban, mc);

		if mvs.len() == 0 {
			return Evaluation::Result(Score::NEGINFINITE,None);
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

		if limit - Instant::now() <= Duration::from_millis(10) {
			return Evaluation::Timeout;
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
			if limit - Instant::now() <= Duration::from_millis(10) {
				return Evaluation::Timeout;
			}
			let empmc = HashMap::new();

			let mp = match mc {
				&MochigomaCollections::Pair(ref s,ref g) => {
					match teban {
						Teban::Sente => s,
						Teban::Gote => g
					}
				},
				&MochigomaCollections::Empty => &empmc,
			};

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
								self.calc_main_hash(mhash,banmen,mp,&m,&o),
								self.calc_sub_hash(shash,banmen,mp,&m,&o)
							)
						},
						LegalMove::Put(ref k, ref d) => {
							let m = Move::Put(*k,*d);

							(
								self.calc_main_hash(mhash,banmen,mp,&m,&None),
								self.calc_sub_hash(shash,banmen,mp,&m,&None)
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
						Some(_) => {
							continue;
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
							let is_put_fu = match m {
								LegalMove::Put(MochigomaKind::Fu,_) => true,
								_ => false,
							};

							match self.respond_oute_only(teban.opposite(),next,mc,
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

		for m in &mvs {
			if limit - Instant::now() <= Duration::from_millis(10) {
				return Evaluation::Timeout;
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
							match self.alphabeta(teban.opposite(),&banmen,
								-beta,-alpha,m.to_move(),&mc,
								obtained,&current_kyokumen_hash_map,
								already_oute_hash_map,mhash,shash,limit,depth-1,current_depth) {

								Evaluation::Timeout => {
									return Evaluation::Timeout;
								},
								Evaluation::Result(s,_) => {
									if -s > scoreval {
										scoreval = -s;
										if alpha < scoreval {
											alpha = scoreval;
										}
										if scoreval >= beta {
											return Evaluation::Result(scoreval,Some(m.to_move()));
										}
									}
								}
							}
						}
					}
				}
			}
		}

		panic!("logic error!");
	}

	fn respond_oute_only(&mut self, teban:Teban,banmen:&Banmen,
								mc:&MochigomaCollections,is_put_fu:bool,
								current_kyokumen_hash_map:&TwoKeyHashMap<u32>,
								already_oute_hash_map:&mut TwoKeyHashMap<()>,
								mhash:u64,shash:u64,
								limit:Instant,current_depth:u32) -> OuteEvaluation {
		let mvs = banmen.respond_oute_only_moves_all(&teban, mc);

		if mvs.len() == 0 {
			return OuteEvaluation::Result(current_depth as i32)
		} else {
			let empmc = HashMap::new();
			let mp = match mc {
				&MochigomaCollections::Pair(ref s,ref g) => {
					match teban {
						Teban::Sente => s,
						Teban::Gote => g
					}
				},
				&MochigomaCollections::Empty => &empmc,
			};

			let mut longest_depth = -1;

			for m in mvs {
				if limit - Instant::now() <= Duration::from_millis(10) {
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
							self.calc_main_hash(mhash,banmen,mp,&m,&o),
							self.calc_sub_hash(shash,banmen,mp,&m,&o)
						)
					},
					LegalMove::Put(ref k, ref d) => {
						let m = Move::Put(*k,*d);

						(
							self.calc_main_hash(mhash,banmen,mp,&m,&None),
							self.calc_sub_hash(shash,banmen,mp,&m,&None)
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
					LegalMove::To(ref s, ref d,_) => {
						banmen.apply_move_none_check(&teban,mc,&Move::To(*s,*d))
					},
					LegalMove::Put(ref k, ref d) => {
						banmen.apply_move_none_check(&teban,mc,&Move::Put(*k,*d))
					}
				};

				match next {
					(ref next,ref mc,_) => {
						match self.oute_only(teban.opposite(),next,mc,
												is_put_fu,current_kyokumen_hash_map,
												already_oute_hash_map,mhash,shash,limit,current_depth+1) {
							OuteEvaluation::Result(d) if d >= 0 && !is_put_fu => {
								return OuteEvaluation::Result(d);
							},
							OuteEvaluation::Result(d) if d >= 0 => {
								longest_depth = longest_depth.max(d);
							},
							OuteEvaluation::Result(d) => {
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

	fn oute_only(&mut self, teban:Teban,banmen:&Banmen,
								mc:&MochigomaCollections,is_put_fu:bool,
								current_kyokumen_hash_map:&TwoKeyHashMap<u32>,
								already_oute_hash_map:&mut TwoKeyHashMap<()>,
								mhash:u64,shash:u64,
								limit:Instant,current_depth:u32) -> OuteEvaluation {
		let mvs = banmen.oute_only_moves_all(&teban, mc);

		if mvs.len() == 0 {
			OuteEvaluation::Result(-1)
		} else {
			let empmc = HashMap::new();
			let mp = match mc {
				&MochigomaCollections::Pair(ref s,ref g) => {
					match teban {
						Teban::Sente => s,
						Teban::Gote => g
					}
				},
				&MochigomaCollections::Empty => &empmc,
			};

			let mut shortest_depth = -1;

			for m in mvs {
				if limit - Instant::now() <= Duration::from_millis(10) {
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
							self.calc_main_hash(mhash,banmen,mp,&m,&o),
							self.calc_sub_hash(shash,banmen,mp,&m,&o)
						)
					},
					LegalMove::Put(ref k, ref d) => {
						let m = Move::Put(*k,*d);

						(
							self.calc_main_hash(mhash,banmen,mp,&m,&None),
							self.calc_sub_hash(shash,banmen,mp,&m,&None)
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
						match self.respond_oute_only(teban.opposite(),next,mc,
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

	fn calc_hash<AF,PF>(&self,h:u64,b:&Banmen,mc:&HashMap<MochigomaKind,u32>,
												m:&Move,obtained:&Option<MochigomaKind>,add:AF,pull:PF)
		-> u64 where AF: Fn(u64,u64) -> u64, PF: Fn(u64,u64) -> u64 {
		match b {
			&Banmen(ref kinds) => {
				match m {
					&Move::To(ref ms, KomaDstToPosition(dx, dy, mn)) => {
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
								let c = match mc.get(obtained) {
									Some(c) => *c as usize,
									None => 0,
								};

								let k = *obtained as usize;

								hash = add(hash,self.mochigoma_hash_seeds[c][k]);
								hash
							}
						};

						hash
					},
					&Move::Put(ref mk, ref md) => {
						let mut hash = h;
						let c = match mc.get(&mk) {
							Some(c) => (*c - 1) as usize,
							None => {
								return hash;
							}
						};
						let k = *mk as usize;

						hash = pull(hash,self.mochigoma_hash_seeds[c][k]);

						let dx = md.0 as usize;
						let dy = md.1 as usize;

						hash = add(hash,self.kyokumen_hash_seeds[k + 1usize][dx * 9 + dy]);
						hash
					}
				}
			}
		}
	}

	fn calc_main_hash(&self,h:u64,b:&Banmen,mc:&HashMap<MochigomaKind,u32>,m:&Move,obtained:&Option<MochigomaKind>) -> u64 {
		self.calc_hash(h,b,mc,m,obtained,|h,v| h ^ v, |h,v| h ^ v)
	}

	fn calc_sub_hash(&self,h:u64,b:&Banmen,mc:&HashMap<MochigomaKind,u32>,m:&Move,obtained:&Option<MochigomaKind>) -> u64 {
		self.calc_hash(h,b,mc,m,obtained,|h,v| {
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
	fn set_option(&mut self,name:String,value:SysEventOption) -> Result<(),CommonError> {
		Ok(())
	}
	fn newgame(&mut self) -> Result<(),CommonError> {
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
		Ok(BestMove::Win)
	}
	fn think_mate<L>(&mut self,limit:&UsiGoMateTimeLimit,event_queue:Arc<Mutex<EventQueue<UserEvent,UserEventKind>>>,
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
		Ok(())
	}

	fn quit(&mut self) -> Result<(),CommonError> {
		Ok(())
	}
}
