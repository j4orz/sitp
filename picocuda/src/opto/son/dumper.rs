use std::{collections::{HashMap}, fmt::{self, Write}};
use crate::opto::son::{parser::{ParseResult, Scope}, DefEdge, OpCode};

pub fn dump_dot(char_input: &[char], graph_output: &ParseResult) -> Result<String, fmt::Error> {
    let mut dump = String::new();
    write!(dump, "digraph {{\n")?;
    write!(dump, "/*\n")?; write!(dump, "{}", char_input.iter().collect::<String>())?; write!(dump, "\n*/\n")?;
    write!(dump, "\trankdir=BT;\n")?; // force nodes before scopes
    write!(dump, "\tordering=\"in\";\n")?; // preserve node input order
    write!(dump, "\tconcentrate=\"true\";\n")?; // merge multiple edges hitting the same node
    let flat_graph = flatten(graph_output);

    dump_nodes(&mut dump, &flat_graph)?;
    dump_scope(&mut dump, &graph_output.scope)?;
    dump_node_edges(&mut dump, &flat_graph)?;
    // dump_scope_edges(&mut dump, &parser.scope)?;
    write!(dump, "}}\n")?;
    Ok(dump)
}

fn dump_nodes(s: &mut String, flat_graph: &Vec<DefEdge>) -> fmt::Result {
    write!(s, "\tsubgraph cluster_Nodes {{\n")?;
    for node in flat_graph {
        if let OpCode::Scope = node.borrow().opcode { continue }
        write!(s, "\t\t{} [ ", node.unique_label())?;
        if node.is_cfg() { write!(s, "shape=box style=filled fillcolor=yellow ")?; } // default is ellipse
        write!(s, "label=\"{:?}\" ", node)?;
        write!(s, "];\n")?;
    }
    write!(s, "\t}}\n")
}
fn dump_node_edges(s: &mut String, flat_graph: &Vec<DefEdge>) -> fmt::Result {
    write!(s, "\tedge [ fontname=Helvetica, fontsize=8 ];\n")?;
    for node in flat_graph {
        if let OpCode::Scope = node.borrow().opcode { continue }
        for (i, def) in (&node.borrow().defs).iter().enumerate() {
            write!(s, "\t{} -> {}", node.unique_label(), def.unique_label())?; // unique labels for DOT (display not enough)
            write!(s, "[taillabel={i}")?;
            if let (OpCode::Con, OpCode::Start) = (node.borrow().opcode, def.borrow().opcode) { write!(s, " style=dotted")?; }
            else if def.is_cfg() { write!(s, " color=red")?; }
            write!(s, "];\n")?;
        }
    }
    Ok(())
}

fn build_scopename() -> String { todo!() }
fn build_portname() -> String { todo!() }
fn dump_scope(s: &mut String, scope: &Scope) -> fmt::Result {
    write!(s, "\tnode [shape=plaintext];\n")?;
    for (i, nv) in (&scope.nvs).iter().enumerate() {
        let scope_name = "";
        write!(s, "\tsubgraph cluster_{scope_name} {{\n")?;
        write!(s, "\t\t{scope_name} [label=<\n")?;

        write!(s, "\t\t\t<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">\n");
        write!(s, "\t\t\t<TR><TD BGCOLOR=\"cyan\">{i}</TD>")?; // add scope level
        // for alias in nv.keys() {write !(s, "<TD PORT=\"{port_name}\">{alias}</TD>")?; }
        write!(s, "</TR>\n");
        write!(s, "\t\t\t</TABLE>>];\n");
    }
    // write!(s, "\t}\n".repeat( level ) ); // End all Scope clusters
    Ok(())
}
fn _dump_scope_edges(s: &mut String, scope: &Scope) -> fmt::Result { todo!() }

fn flatten(parsed: &ParseResult) -> Vec<DefEdge> {
    let mut seen = HashMap::new();
    traverse_graph_from_node(&parsed.start, &mut seen);
    traverse_scope_bindings(&parsed.scope, &mut seen);
    seen.values().cloned().collect::<Vec<_>>()
}

fn traverse_scope_bindings(scope: &Scope, seen: &mut HashMap<usize, DefEdge>) -> () {
    for nv in &scope.nvs {
        for idef in nv.values() { // NB: linear in the size of bindings
            let bound_expr = &scope.lookup.borrow().defs[*idef];
            traverse_graph_from_node(bound_expr, seen)
        }
    }
}

fn traverse_graph_from_node(node: &DefEdge, seen: &mut HashMap<usize, DefEdge>) -> () {
    if seen.contains_key(&node.borrow().id) { return } else { seen.insert(node.borrow().id, node.clone());}
    for n in &node.borrow().defs { traverse_graph_from_node(node, seen); }
    for n in &node.borrow().uses { traverse_graph_from_node(&DefEdge::from_upgraded(n.upgrade().unwrap()), seen) }
}

// fn bfs(root: &DefEdge, max_depth: usize) -> (Vec<DefEdge>, Vec<bool>) {
//     let (mut queue, mut seen) = (VecDeque::new(), Vec::new());
//     let (i, depth) = (0, 0);
//     let (_, _) = (queue.push_back(root), seen[root.0.borrow().id] = true);

//     todo!()
// }

// impl Debug for DefEdge {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let (bfs, seen) = bfs(self, 9999);
//         // f.debug_tuple("DefEdge").field(&self.0).finish()
//         todo!()
//     }
// }