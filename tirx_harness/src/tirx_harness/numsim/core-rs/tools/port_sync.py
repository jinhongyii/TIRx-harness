import re, sys, os
REF = sys.argv[1]; DST = sys.argv[2]
KEEP = {"arrival_limit", "default_count"}
STUB = {"step","work_step","quiescent","check_invariants","enabled_grants","wait_prefix_len","initial_total"}
for mod in ["mbarrier","named","cluster","async_group","tcgen","setmaxnreg"]:
    lines = open(os.path.join(REF, mod + ".rs")).read().split("\n")
    out = []; i = 0
    # header doc
    doc = []
    while i < len(lines) and lines[i].startswith("//!"):
        doc.append(lines[i]); i += 1
    out += doc
    out += ["//!",
            "//! CONTRACT: the types in this file are copied verbatim from",
            "//! `numsim-sync-ref/src/%s.rs` (plus serde derives) so production `step`" % mod,
            "//! is differentially tested against the reference mechanically. Change",
            "//! them only together with the reference crate, via the coordinator.",
            "//! Function bodies marked `W3` are the production implementation to write."]
    while i < len(lines):
        l = lines[i]
        m = re.match(r'^(pub )?fn (\w+)', l)
        if m and l.startswith(("fn ", "pub fn ")):
            # collect fn through closing '}' at col 0
            j = i
            while lines[j] != "}":
                j += 1
            name = m.group(2)
            # strip preceding doc comments for private fns
            if not m.group(1):
                while out and (out[-1].startswith("///") or out[-1].startswith("#[")):
                    out.pop()
                i = j + 1
                continue
            if name not in KEEP:
                # signature lines up to '{'
                k = i
                sig = []
                while True:
                    sig.append(lines[k])
                    if lines[k].rstrip().endswith("{"):
                        break
                    k += 1
                out += sig
                out.append('    unimplemented!("W3: %s::%s")' % (mod, name))
                out.append("}")
                i = j + 1
                continue
        if l.startswith("#[cfg(test)]"):
            break
        l = l.replace("use crate::", "use super::").replace("impl crate::Protocol", "impl super::Protocol")
        if l.startswith("#[derive(") and "Debug" in l:
            l = l.replace(")]", ", serde::Serialize, serde::Deserialize)]")
        out.append(l)
        i += 1
    # suppress unused-arg warnings in stubs
    text = "\n".join(out).rstrip() + "\n"
    text = text.replace("//!\n", "//!\n", 1)
    if "pub fn quiescent" not in text:
        text += "\n/// End-of-launch check (contract addition; not yet in the reference).\npub fn quiescent(s: &State) -> Result<(), Error> {\n    unimplemented!(\"W3: %s::quiescent\")\n}\n" % mod
    open(os.path.join(DST, mod + ".rs"), "w").write("#![allow(unused_variables, dead_code)]\n" + text)
