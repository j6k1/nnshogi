use std::fmt;
use std::error;
use std::io;
use std::convert::From;
use usiagent::event::SystemEventKind;
use usiagent::event::UserEventKind;
use usiagent::error::USIAgentRunningError;
use usiagent::error::USIAgentStartupError;
use usiagent::error::PlayerError;
use usiagent::error::UsiProtocolError;
use simplenn::error::InvalidStateError;
use simplenn::error::PersistenceError;

#[derive(Debug)]
pub enum CommonError {
	Fail(String),
}
impl PlayerError for CommonError {

}
impl fmt::Display for CommonError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			CommonError::Fail(ref s) => write!(f, "{}",s),
		}
	}
}
impl error::Error for CommonError {
	fn description(&self) -> &str {
		match *self {
			CommonError::Fail(_) => "Player error.",
		}
	}

	fn cause(&self) -> Option<&error::Error> {
		match *self {
			CommonError::Fail(_) => None,
		}
	}
}
impl<'a> From<CommonError> for USIAgentRunningError<'a,SystemEventKind,CommonError>
	where SystemEventKind: fmt::Debug {
	fn from(err: CommonError) -> USIAgentRunningError<'a,SystemEventKind,CommonError> {
		USIAgentRunningError::from(USIAgentStartupError::PlayerError(err))
	}
}
impl<'a> From<CommonError> for USIAgentRunningError<'a,UserEventKind,CommonError>
	where UserEventKind: fmt::Debug {
	fn from(err: CommonError) -> USIAgentRunningError<'a,UserEventKind,CommonError> {
		USIAgentRunningError::from(USIAgentStartupError::PlayerError(err))
	}
}
impl From<InvalidStateError> for CommonError {
	fn from(_: InvalidStateError) -> CommonError {
		CommonError::Fail(String::from("invalid state."))
	}
}
impl From<io::Error> for CommonError {
	fn from(_: io::Error) -> CommonError {
		CommonError::Fail(String::from("I/O Error."))
	}
}
impl<E> From<PersistenceError<E>> for CommonError {
	fn from(_: PersistenceError<E>) -> CommonError {
		CommonError::Fail(String::from("An error occurred while saving model of NN."))
	}
}

impl From<UsiProtocolError> for CommonError {
	fn from(err: UsiProtocolError) -> CommonError {
		match err {
			UsiProtocolError::InvalidState(s) => CommonError::Fail(s)
		}
	}
}
#[derive(Debug)]
pub enum ApplicationError {
	StartupError(String),
}
impl fmt::Display for ApplicationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			ApplicationError::StartupError(ref s) => write!(f, "{}",s),
		}
	}
}
impl error::Error for ApplicationError {
	fn description(&self) -> &str {
		match *self {
			ApplicationError::StartupError(_) => "Startup failed.",
		}
	}

	fn cause(&self) -> Option<&error::Error> {
		match *self {
			ApplicationError::StartupError(_) => None,
		}
	}
}