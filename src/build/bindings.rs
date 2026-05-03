use super::reflection::{Param, Reflection};

pub fn generate_bindings(reflection: &Reflection, target_name: &str) -> String {
	let mut out = String::new();

	for ep in &reflection.entry_points {
		out.push_str(&format!(
			"pub const {target_name}_ENTRY_NAME: &str = \"{}\";\n",
			ep.name
		));
		out.push_str(&format!(
			"pub const {target_name}_ENTRY_STAGE: &str = \"{}\";\n",
			ep.stage
		));
		out.push_str(&format!(
			"pub const {target_name}_THREAD_GROUP: [u64; 3] = [{}, {}, {}];\n",
			ep.thread_group_size[0], ep.thread_group_size[1], ep.thread_group_size[2],
		));

		let bindable: Vec<&Param> = ep
			.parameters.iter()
			.filter(|p| p.binding.as_ref().map_or(false, |b| !b.kind.is_empty()))
			.collect();

		out.push_str(&format!(
			"pub const {target_name}_PARAM_COUNT: usize = {};\n",
			bindable.len()
		));

		for (i, param) in bindable.iter().enumerate() {
			let b = param.binding.as_ref().unwrap();
			let upper = param.name.to_uppercase();

			out.push_str(&format!(
				"pub const {target_name}_P{i}_{upper}_NAME: &str = \"{}\";\n",
				param.name
			));
			out.push_str(&format!(
				"pub const {target_name}_P{i}_{upper}_KIND: &str = \"{}\";\n",
				b.kind
			));

			if let Some(idx) = b.index {
				out.push_str(&format!("pub const {target_name}_P{i}_{upper}_INDEX: u32 = {idx};\n"));
			}
			if let Some(offset) = b.offset {
				out.push_str(&format!("pub const {target_name}_P{i}_{upper}_OFFSET: usize = {offset};\n"));
			}
			if let Some(size) = b.size {
				out.push_str(&format!("pub const {target_name}_P{i}_{upper}_SIZE: usize = {size};\n"));
			}

			if b.kind == "constantBuffer" {
				if let Some(inner) = &param.ty.element_type {
					if let Some(fields) = &inner.fields {
						for field in fields {
							let fu = field.name.to_uppercase();
							out.push_str(&format!(
								"pub const {target_name}_P{i}_{upper}_FIELD_{fu}_OFFSET: usize = {};\n",
								field.binding.offset.unwrap_or(0)
							));
							out.push_str(&format!(
								"pub const {target_name}_P{i}_{upper}_FIELD_{fu}_SIZE: usize = {};\n",
								field.binding.size.unwrap_or(0)
							));
						}
					}
				}
			}
		}

		if target_name == "CPU" {
			out.push('\n');
			generate_cpu_ep_layout(&mut out, &bindable);
		}
	}

	out
}

fn generate_cpu_ep_layout(out: &mut String, params: &[&Param]) {
	let total: u64 = params.iter()
		.filter_map(|p| p.binding.as_ref().unwrap().size)
		.sum();
	out.push_str(&format!("pub const CPU_EP_PARAMS_TOTAL_SIZE: usize = {total};\n"));

	for param in params {
		let b = param.binding.as_ref().unwrap();
		let upper = param.name.to_uppercase();
		let offset = b.offset.unwrap_or(0);
		let size = b.size.unwrap_or(0);

		let rust_type = if b.kind == "uniform" {
			match size {
				16 => if param.ty.access.as_deref() == Some("readWrite") { "RwBufU32" } else { "RoBufU32" },
				8 => "PtrConst",
				_ => "Unknown",
			}
		} else {
			"Unknown"
		};

		out.push_str(&format!("pub const CPU_EP_{upper}_OFFSET: usize = {offset};\n"));
		out.push_str(&format!("pub const CPU_EP_{upper}_SIZE: usize = {size};\n"));
		out.push_str(&format!("pub const CPU_EP_{upper}_TYPE: &str = \"{rust_type}\";\n"));
	}
}
