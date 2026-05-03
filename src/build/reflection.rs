use serde::Deserialize;

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct Reflection {
	#[serde(default)]
	pub parameters: Vec<Param>,
	#[serde(rename = "entryPoints")]
	pub entry_points: Vec<EntryPoint>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct EntryPoint {
	pub name: String,
	pub stage: String,
	#[serde(default)]
	pub parameters: Vec<Param>,
	#[serde(rename = "threadGroupSize")]
	pub thread_group_size: [u64; 3],
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct Param {
	pub name: String,
	#[serde(default, rename = "semanticName")]
	pub semantic_name: Option<String>,
	#[serde(default)]
	pub binding: Option<Binding>,
	#[serde(rename = "type")]
	pub ty: Type,
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)]
pub struct Binding {
	#[serde(default)]
	pub kind: String,
	#[serde(default)]
	pub index: Option<u64>,
	#[serde(default)]
	pub offset: Option<u64>,
	#[serde(default)]
	pub size: Option<u64>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct Type {
	#[serde(default)]
	pub kind: String,
	#[serde(default, rename = "baseShape")]
	pub base_shape: Option<String>,
	#[serde(default, rename = "access")]
	pub access: Option<String>,
	#[serde(default, rename = "elementType")]
	pub element_type: Option<Box<Type>>,
	#[serde(default, rename = "elementCount")]
	pub element_count: Option<u64>,
	#[serde(default, rename = "scalarType")]
	pub scalar_type: Option<String>,
	#[serde(default, rename = "name")]
	pub type_name: Option<String>,
	#[serde(default, rename = "fields")]
	pub fields: Option<Vec<Field>>,
	#[serde(default, rename = "resultType")]
	pub result_type: Option<Box<Type>>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct Field {
	pub name: String,
	#[serde(rename = "type")]
	pub ty: Type,
	pub binding: Binding,
}

pub fn parse_reflection(json: &str) -> Result<Reflection, serde_json::Error> {
	serde_json::from_str(json)
}
