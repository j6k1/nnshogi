use std::fmt;
use std::error;
use std::io;
use std::convert::From;
use std::num::ParseIntError;
use std::num::ParseFloatError;
use usiagent::event::SystemEventKind;
use usiagent::event::UserEventKind;
use usiagent::error::USIAgentRunningError;
use usiagent::error::USIAgentStartupError;
use usiagent::error::PlayerError;
use usiagent::error::UsiProtocolError;
use usiagent::error::SelfMatchRunningError;
use usiagent::error::TypeConvertError;
use simplenn::error::InvalidStateError;
use simplenn::error::PersistenceError;
use usiagent::error::SfenStringConvertError;
use usiagent::error::KifuWriteError;
use csaparser::error::CsaParserError;

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

	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
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
	fn from(err: InvalidStateError) -> CommonError {
		CommonError::Fail(format!("invalid state. ({})",err))
	}
}
impl From<TypeConvertError<String>> for CommonError {
	fn from(err: TypeConvertError<String>) -> CommonError {
		CommonError::Fail(format!("An error occurred during type conversion. ({})",err))
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
	SfenStringConvertError(SfenStringConvertError),
	IOError(io::Error),
	ParseIntError(ParseIntError),
	ParseFloatError(ParseFloatError),
	AgentRunningError(String),
	SelfMatchRunningError(SelfMatchRunningError<CommonError>),
	CsaParserError(CsaParserError),
	LogicError(String),
	LearningError(String),
	KifuWriteError(KifuWriteError),
	SerdeError(toml::ser::Error)
}
impl fmt::Display for ApplicationError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match *self {
			ApplicationError::StartupError(ref s) => write!(f, "{}",s),
			ApplicationError::SfenStringConvertError(ref e) => write!(f, "{}",e),
			ApplicationError::IOError(ref e) => write!(f, "{}",e),
			ApplicationError::ParseIntError(ref e) => write!(f, "{}",e),
			ApplicationError::ParseFloatError(ref e) => write!(f, "{}",e),
			ApplicationError::AgentRunningError(ref s) => write!(f, "{}",s),
			ApplicationError::SelfMatchRunningError(ref e) => write!(f, "{}",e),
			ApplicationError::CsaParserError(ref e) => write!(f, "{}",e),
			ApplicationError::LogicError(ref s) => write!(f,"{}",s),
			ApplicationError::LearningError(ref s) => write!(f,"{}",s),
			ApplicationError::KifuWriteError(ref s) => write!(f,"{}",s),
			ApplicationError::SerdeError(ref e) => write!(f,"{}",e),
		}
	}
}
impl error::Error for ApplicationError {
	fn description(&self) -> &str {
		match *self {
			ApplicationError::StartupError(_) => "Startup Error.",
			ApplicationError::SfenStringConvertError(_) => "An error occurred during conversion to sfen string.",
			ApplicationError::IOError(_) => "IO Error.",
			ApplicationError::ParseIntError(_) => "An error occurred parsing the integer string.",
			ApplicationError::ParseFloatError(_) => "An error occurred parsing the float string.",
			ApplicationError::AgentRunningError(_) => "An error occurred while running USIAgent.",
			ApplicationError::SelfMatchRunningError(_) => "An error occurred while running the self-match.",
			ApplicationError::CsaParserError(_) => "An error occurred parsing the csa file.",
			ApplicationError::LogicError(_) => "Logic error.",
			ApplicationError::LearningError(_) => "An error occurred while learning the neural network.",
			ApplicationError::KifuWriteError(_) => "An error occurred when recording kifu or initialize KifuWriter.",
			ApplicationError::SerdeError(_) => "An error occurred during serialization or deserialization."
		}
	}

	fn source(&self) -> Option<&(dyn error::Error + 'static)> {
		match *self {
			ApplicationError::StartupError(_) => None,
			ApplicationError::SfenStringConvertError(ref e) => Some(e),
			ApplicationError::IOError(ref e) => Some(e),
			ApplicationError::ParseIntError(ref e) => Some(e),
			ApplicationError::ParseFloatError(ref e) => Some(e),
			ApplicationError::AgentRunningError(_) => None,
			ApplicationError::SelfMatchRunningError(ref e) => Some(e),
			ApplicationError::CsaParserError(ref e) => Some(e),
			ApplicationError::LogicError(_) => None,
			ApplicationError::LearningError(_) => None,
			ApplicationError::KifuWriteError(ref e) => Some(e),
			ApplicationError::SerdeError(ref e) => Some(e),
		}
	}
}
impl From<io::Error> for ApplicationError {
	fn from(err: io::Error) -> ApplicationError {
		ApplicationError::IOError(err)
	}
}
impl From<ParseIntError> for ApplicationError {
	fn from(err: ParseIntError) -> ApplicationError {
		ApplicationError::ParseIntError(err)
	}
}
impl From<ParseFloatError> for ApplicationError {
	fn from(err: ParseFloatError) -> ApplicationError {
		ApplicationError::ParseFloatError(err)
	}
}
impl From<SelfMatchRunningError<CommonError>> for ApplicationError {
	fn from(err: SelfMatchRunningError<CommonError>) -> ApplicationError {
		ApplicationError::SelfMatchRunningError(err)
	}
}
impl From<SfenStringConvertError> for ApplicationError {
	fn from(err: SfenStringConvertError) -> ApplicationError {
		ApplicationError::SfenStringConvertError(err)
	}
}
impl From<KifuWriteError> for ApplicationError {
	fn from(err: KifuWriteError) -> ApplicationError {
		ApplicationError::KifuWriteError(err)
	}
}
impl From<CsaParserError> for ApplicationError {
	fn from(err: CsaParserError) -> ApplicationError {
		ApplicationError::CsaParserError(err)
	}
}
impl From<toml::ser::Error> for ApplicationError {
	fn from(err: toml::ser::Error) -> ApplicationError {
		ApplicationError::SerdeError(err)
	}
}
