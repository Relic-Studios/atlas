"""Codex account transport: ephemeral, language-only threads and Atlas-owned tools."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from codex_rpc import CodexRPC, DISABLED_FEATURES, executable
from provider_errors import ProviderError, failure

SUPPORTED_VERSION = "0.146.1"


def _toml(value):
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k) + "=" + _toml(v) for k,v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_toml(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=False)


def verify_config(cfg):
    if (cfg.get("model_provider") != "openai" or cfg.get("approval_policy") != "never"
            or cfg.get("sandbox_mode") != "read-only"
            or cfg.get("openai_base_url") or cfg.get("chatgpt_base_url")
            or cfg.get("features", {}).get("view_image") is not False
            or cfg.get("web_search") != "disabled" or cfg.get("project_doc_max_bytes") != 0
            or cfg.get("skills", {}).get("include_instructions") is not False
            or cfg.get("skills", {}).get("bundled", {}).get("enabled") is not False
            or any(v.get("enabled") is not False for v in cfg.get("mcp_servers", {}).values())
            or any(v.get("enabled") is not False for v in cfg.get("plugins", {}).values())):
        raise ProviderError("codex", "isolation", "Inherited Codex capabilities or service endpoints could not be isolated. No inference was started.")


def isolated_rpc(configured=None, cancel_event=None):
    version = subprocess.run([executable("codex", configured), "--version"], capture_output=True,
        text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
    if version != "codex-cli " + SUPPORTED_VERSION:
        raise ProviderError("codex", "incompatible_cli", "This Codex CLI version has not passed Atlas's protocol checks. Use the supported version or update Atlas.")
    # Discover paths without executing skills, then disable every discovered path in the inference child.
    with CodexRPC(configured, cancel_event=cancel_event) as discovery:
        listed = discovery.call("skills/list", {"cwds": [discovery.cwd], "forceReload": True})
    paths = []
    for group in listed.get("data", []):
        if group.get("errors"):
            raise ProviderError("codex", "isolation", "Codex skills could not be inspected safely.")
        paths.extend({"path": s["path"], "enabled": False} for s in group.get("skills", []))
    rpc = CodexRPC(configured, extra_overrides=["-c", "skills.config=" + _toml(paths)], cancel_event=cancel_event)
    try:
        cfg = rpc.call("config/read", {"includeLayers": False}).get("config", {})
        verify_config(cfg)
        features, cursor = {}, None
        for _ in range(20):
            page = rpc.call("experimentalFeature/list", {"limit": 200, "cursor": cursor})
            features.update({v["name"]: v["enabled"] for v in page.get("data", [])})
            cursor = page.get("nextCursor")
            if not cursor:
                break
        # view_image is a stable feature in this pinned CLI, verified in config/read above.
        if any(features.get(name) is not False for name in DISABLED_FEATURES if name != "view_image"):
            raise ProviderError("codex", "isolation", "Codex's restricted feature configuration could not be verified.")
        listed = rpc.call("skills/list", {"cwds": [rpc.cwd], "forceReload": True})
        if any(g.get("errors") or any(s.get("enabled") is not False for s in g.get("skills", [])) for g in listed.get("data", [])):
            raise ProviderError("codex", "isolation", "Codex still has an enabled skill. No inference was started.")
        return rpc
    except BaseException:
        rpc.close()
        raise


def instruction_fingerprint():
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    digest = hashlib.sha256()
    for name in ("AGENTS.override.md", "AGENTS.md"):
        p = home / name
        if p.is_file():
            digest.update(name.encode())
            digest.update(p.read_bytes())
            break
    return digest.hexdigest()


def tool_specs(tools):
    specs = {}
    for item in tools or []:
        f = item.get("function", {})
        name = f.get("name", "")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", name) or name in specs:
            raise ProviderError("codex", "tool_schema", "An Atlas tool has an unsupported or duplicate name.")
        specs[name] = {"name": name, "description": f.get("description", ""),
                       "inputSchema": f.get("parameters", {"type": "object", "properties": {}})}
    return specs


def valid_arguments(value, schema, depth=0):
    """Validate the bounded JSON-schema subset used by Atlas tools; unknown keywords fail closed."""
    if depth > 20:
        return False
    if isinstance(value,float) and not math.isfinite(value):
        return False
    annotations = {"description", "title", "default", "examples", "$schema"}
    supported = annotations | {"type", "properties", "required", "additionalProperties", "items", "enum", "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "anyOf", "oneOf", "nullable"}
    if any(k not in supported for k in schema):
        return False
    if schema.get("nullable") and value is None:
        return True
    if "anyOf" in schema and not any(valid_arguments(value,s,depth+1) for s in schema["anyOf"]):
        return False
    if "oneOf" in schema and sum(valid_arguments(value,s,depth+1) for s in schema["oneOf"]) != 1:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    kind = schema.get("type")
    types = {"object": lambda v:isinstance(v,dict), "array":lambda v:isinstance(v,list),
             "string":lambda v:isinstance(v,str), "boolean":lambda v:isinstance(v,bool),
             "number":lambda v:isinstance(v,(float,int)) and not isinstance(v,bool),
             "integer":lambda v:isinstance(v,int) and not isinstance(v,bool), "null":lambda v:v is None}
    if kind:
        choices = kind if isinstance(kind,list) else [kind]
        if not any(k in types and types[k](value) for k in choices):
            return False
    if isinstance(value,dict):
        props = schema.get("properties", {})
        if any(k not in value for k in schema.get("required", [])):
            return False
        for k,v in value.items():
            if k in props and not valid_arguments(v,props[k],depth+1):
                return False
            if k not in props:
                extra = schema.get("additionalProperties", True)
                if extra is False or isinstance(extra,dict) and not valid_arguments(v,extra,depth+1):
                    return False
    if isinstance(value,list):
        if len(value)<schema.get("minItems",0) or len(value)>schema.get("maxItems",10000):
            return False
        if "items" in schema and not all(valid_arguments(v,schema["items"],depth+1) for v in value):
            return False
    if isinstance(value,str) and (len(value)<schema.get("minLength",0) or len(value)>schema.get("maxLength",1_000_000)):
        return False
    if isinstance(value,(float,int)) and not isinstance(value,bool):
        if value<schema.get("minimum",float('-inf')) or value>schema.get("maximum",float('inf')):
            return False
    return True


def history_items(messages):
    system, items, pending = [], [], []
    for i,m in enumerate(messages):
        role = m.get("role")
        text = m.get("content")
        if text is None:
            text = ""
        if not isinstance(text,str):
            raise ProviderError("codex", "protocol", "Atlas supplied an unsupported message shape.")
        if role in ("system", "developer"):
            system.append(text)
        elif role == "tool":
            cid = m.get("tool_call_id")
            if not cid:
                match = next((p for p in pending if p[1]==m.get("name") or p[1]==m.get("tool_name")), None)
                if match is None and len(pending)==1:
                    match=pending[0]
                if match:
                    cid=match[0]
            if not cid or not any(p[0]==cid for p in pending):
                raise ProviderError("codex", "protocol", "A tool result has no matching Atlas call.")
            pending=[p for p in pending if p[0]!=cid]
            output = text
            if m.get("images"):
                from remote_llm import _image_part
                output = [{"type":"input_text","text":text}]
                for image in m["images"]:
                    if not isinstance(image,str):
                        raise ProviderError("codex","protocol","Atlas supplied an unsupported tool image.")
                    output.append({"type":"input_image","image_url":_image_part(image)["image_url"]["url"]})
            items.append({"type":"function_call_output","call_id":cid,"output":output})
        elif role in ("user","assistant"):
            if role == "user" and text.startswith("/no_think "):
                text=text[len("/no_think "):]
            content=[]
            if text:
                content.append({"type":"input_text" if role=="user" else "output_text","text":text})
            for b64 in m.get("images",[]):
                if role!="user" or not isinstance(b64,str):
                    raise ProviderError("codex","protocol","Atlas supplied an unsupported image message.")
                from remote_llm import _image_part
                content.append({"type":"input_image","image_url":_image_part(b64)["image_url"]["url"]})
            if content:
                items.append({"type":"message","role":role,"content":content})
            for n,call in enumerate(m.get("tool_calls") or []):
                f=call.get("function",{})
                cid=call.get("id") or f"atlas-{i}-{n}"
                args=f.get("arguments",{})
                if not isinstance(args,str):
                    args=json.dumps(args,ensure_ascii=False)
                items.append({"type":"function_call","call_id":cid,"name":f.get("name"),"arguments":args})
                pending.append((cid,f.get("name")))
        else:
            raise ProviderError("codex","protocol","Atlas supplied an unsupported message role.")
    if pending:
        raise ProviderError("codex","protocol","Atlas supplied an incomplete tool exchange.")
    return "\n\n".join(system),items


def generate(cfg, payload, response):
    rpc=isolated_rpc(cfg.get("executable"), response.cancelled)
    response.session.isolation_verified=True
    response.session.cli_version=SUPPORTED_VERSION
    thread_id=turn_id=None
    try:
        def cancel():
            try:
                if thread_id and turn_id:
                    rpc.call("turn/interrupt",{"threadId":thread_id,"turnId":turn_id},timeout=2)
            except Exception:
                pass
            finally:
                rpc.close()
        response.set_cancel_hook(cancel)
        if response.cancelled.is_set():
            return
        account=rpc.call("account/read",{"refreshToken":False}).get("account") or {}
        if account.get("type")!="chatgpt":
            raise ProviderError("codex","sign_in_required","Connect a ChatGPT account through Codex. Atlas will not switch to API billing.")
        from provider_accounts import account_binding
        if cfg.get("account_binding") and account_binding("codex",account)!=cfg["account_binding"]:
            response.session.invalidate_account()
            raise ProviderError("codex","account_changed","The Codex account changed. Verify and save this brain again in Settings.")
        current=instruction_fingerprint()
        if cfg.get("instruction_fingerprint") and current!=cfg["instruction_fingerprint"]:
            raise ProviderError("codex","configuration_changed","Codex's global instructions changed. Verify this connector again in Settings.")
        system,items=history_items(payload.get("messages",[]))
        specs=tool_specs(payload.get("tools",[]))
        started=rpc.call("thread/start", {"model":cfg["model"],"modelProvider":"openai",
            "approvalPolicy":"never","sandbox":"read-only","cwd":rpc.cwd,"ephemeral":True,
            "allowProviderModelFallback":False,"environments":[],"baseInstructions":system,
            "developerInstructions":"This is Atlas language inference. Only the explicitly supplied Atlas tools may be requested. Do not use coding tools, skills or machine access. Follow the supplied conversation participation and output format.",
            "dynamicTools":list(specs.values())})
        if started.get("model") and started["model"]!=cfg["model"]:
            raise ProviderError("codex","model_changed","Codex selected a different model. Atlas refused the substitution.")
        response.session.resolved_model=started.get("model") or cfg["model"]
        thread_id=started["thread"]["id"]
        fresh=[]
        if items and items[-1].get("type")=="message" and items[-1].get("role")=="user":
            last=items.pop()
            for part in last["content"]:
                if part["type"]=="input_text":
                    fresh.append({"type":"text","text":part["text"]})
                else:
                    fresh.append({"type":"image","url":part["image_url"]})
        if items:
            rpc.call("thread/inject_items",{"threadId":thread_id,"items":items})
        result=rpc.call("turn/start",{"threadId":thread_id,"input":fresh,"approvalPolicy":"never"})
        turn_id=result["turn"]["id"]
        calls,seen=[],{}
        interrupted=False
        deadline=time.monotonic()+300
        while not response.cancelled.is_set():
            if time.monotonic()>deadline:
                raise ProviderError("codex","timeout","Codex did not finish the turn in time.")
            event=rpc.next_event(.1)
            if not event:
                continue
            method=event.get("method","")
            p=event.get("params",{})
            if "id" in event:
                if (method!="item/tool/call" or p.get("threadId")!=thread_id or p.get("turnId")!=turn_id):
                    rpc.reject(event)
                    raise ProviderError("codex","isolation","Codex requested an operation outside the current Atlas tools.")
                name=p.get("tool")
                args=p.get("arguments")
                cid=p.get("callId")
                if (not isinstance(cid,str) or not cid or name not in specs or not isinstance(args,dict)
                        or not valid_arguments(args,specs[name]["inputSchema"])):
                    rpc.reject(event)
                    raise ProviderError("codex","tool_schema","Codex returned an invalid Atlas tool request.")
                if cid not in seen:
                    seen[cid]=(name,args)
                    calls.append({"id":cid,"type":"function","function":{"name":name,"arguments":args}})
                elif seen[cid]!=(name,args):
                    raise ProviderError("codex","tool_schema","Codex reused a tool call ID with different arguments.")
                if len(calls)>10:
                    raise ProviderError("codex","tool_budget","Codex exceeded the tool callback limit.")
                if not interrupted:
                    interrupted=True
                    rpc.call("turn/interrupt",{"threadId":thread_id,"turnId":turn_id},timeout=2)
                    deadline=min(deadline,time.monotonic()+2)
                continue
            if p.get("threadId")!=thread_id:
                continue
            if p.get("turnId") and p["turnId"]!=turn_id:
                continue
            if method=="item/agentMessage/delta" and not calls:
                delta=p.get("delta","")
                if delta:
                    yield {"message":{"role":"assistant","content":delta},"done":False}
            elif method in ("item/started","item/completed"):
                kind=p.get("item",{}).get("type")
                if kind not in (None,"agentMessage","reasoning","userMessage","dynamicToolCall","plan"):
                    raise ProviderError("codex","isolation","Codex attempted a capability outside Atlas's language adapter.")
            elif method=="turn/completed":
                turn=p.get("turn",{})
                if turn.get("id")!=turn_id:
                    continue
                status=turn.get("status")
                if calls and status=="interrupted":
                    yield {"message":{"role":"assistant","content":"","tool_calls":calls},"done":True}
                    return
                if status!="completed":
                    raise failure("codex",turn.get("error") or status)
                yield {"message":{"role":"assistant","content":""},"done":True}
                return
        return
    finally:
        if thread_id:
            try:
                if turn_id and response.cancelled.is_set():
                    rpc.call("turn/interrupt",{"threadId":thread_id,"turnId":turn_id},timeout=2)
                rpc.call("thread/unsubscribe",{"threadId":thread_id},timeout=2)
            except Exception:
                pass
        rpc.close()
