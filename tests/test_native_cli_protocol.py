"""Opt-in real Claude CLI against a loopback model fixture, never real credentials.

ATLAS_CLI_PROTOCOL_TESTS=1 python -m unittest tests.test_native_cli_protocol
Normal public CI skips this installed-CLI integration; pure contracts always run.
"""
import json
import os
import subprocess
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest.mock import patch

from codex_rpc import child_environment,executable
from native_providers import NativeSession
from provider_errors import ProviderError


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_):pass

    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))))
        if "count_tokens" in self.path:
            self.reply_json(200,{"input_tokens":100})
            return
        self.server.requests.append(body)
        if self.server.mode=="limited":
            self.reply_json(429,{"type":"error","error":{"type":"rate_limit_error","message":"Fixture usage limit reached"}})
            return
        tools=body.get("tools",[])
        self.server.tool_names.extend(t.get("name") for t in tools)
        blocks=[]
        if self.server.mode=="tools":
            name=next(t["name"] for t in tools if t["name"].startswith("mcp__atlas__"))
            blocks=[{"type":"tool_use","id":f"fixture-call-{n}","name":name,"input":{"zone":zone}}
                    for n,zone in enumerate(("Asia/Tokyo","UTC"))]
        else:
            blocks=[{"type":"text","text":"A clear fixture reply."}]
        start={"id":"fixture-msg","type":"message","role":"assistant","content":[],"model":body.get("model"),
               "stop_reason":None,"stop_sequence":None,"usage":{"input_tokens":100,"output_tokens":15}}
        events=[("message_start",{"type":"message_start","message":start})]
        for index,block in enumerate(blocks):
            initial=dict(block)
            if block["type"]=="text":initial["text"]=""
            else:initial["input"]={}
            events.append(("content_block_start",{"type":"content_block_start","index":index,"content_block":initial}))
            delta={"type":"text_delta","text":block["text"]} if block["type"]=="text" else {"type":"input_json_delta","partial_json":json.dumps(block["input"])}
            events.extend([("content_block_delta",{"type":"content_block_delta","index":index,"delta":delta}),
                           ("content_block_stop",{"type":"content_block_stop","index":index})])
        events.extend([("message_delta",{"type":"message_delta","delta":{"stop_reason":"tool_use" if self.server.mode=="tools" else "end_turn","stop_sequence":None},"usage":{"output_tokens":15}}),
                       ("message_stop",{"type":"message_stop"})])
        raw="".join("event: "+name+"\ndata: "+json.dumps(event)+"\n\n" for name,event in events).encode()
        if self.server.mode=="slow":
            self.send_response(200);self.send_header("Content-Type","text/event-stream");self.end_headers()
            prefix="".join("event: "+name+"\ndata: "+json.dumps(event)+"\n\n" for name,event in events[:3]).encode()
            self.wfile.write(prefix);self.wfile.flush()
            self.server.release.wait(10)
            return
        self.send_response(200);self.send_header("Content-Type","text/event-stream");self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)

    def reply_json(self,status,value):
        raw=json.dumps(value).encode()
        self.send_response(status);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)


@unittest.skipUnless(os.environ.get("ATLAS_CLI_PROTOCOL_TESTS")=="1","opt-in real CLI/loopback integration")
class ClaudeCLIProtocol(unittest.TestCase):
    def setUp(self):
        import claude_native
        try:
            version=subprocess.run([executable("claude"),"--version"],capture_output=True,text=True,timeout=10).stdout.strip()
        except ProviderError as error:self.skipTest(str(error))
        if version!=claude_native.SUPPORTED_VERSION+" (Claude Code)":self.skipTest("CLI version differs from the supported fixture")
        self.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        self.server.requests=[];self.server.tool_names=[];self.server.mode="text"
        self.server.release=threading.Event()
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        env=child_environment("claude")
        env.update(ANTHROPIC_API_KEY="atlas-fixture-not-a-key",ANTHROPIC_BASE_URL=f"http://127.0.0.1:{self.server.server_port}")
        self.auth=patch("claude_native.account_status",return_value={"connected":True,"account_binding":"fixture-account"})
        self.environment=patch("claude_native.child_environment",return_value=env)
        self.auth.start();self.environment.start()
        self.session=NativeSession({"provider":"claude_code","model":"sonnet","label":"Loopback fixture","url":"","account_binding":"fixture-account"})

    def tearDown(self):
        self.session.close();self.auth.stop();self.environment.stop()
        self.server.release.set()
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)

    def request(self,messages,tools=None):
        return self.session.post("/api/chat",json={"messages":messages,**({"tools":tools} if tools else {})}).json()

    def test_text_and_followup_history_make_only_one_request(self):
        result=self.request([{"role":"system","content":"Fixture persona."},{"role":"user","content":"PAST_USER"},
            {"role":"assistant","content":"PAST_ASSISTANT"},{"role":"user","content":"CURRENT_USER"}])
        self.assertEqual(result["message"]["content"],"A clear fixture reply.")
        self.assertEqual(len(self.server.requests),1)
        wire=json.dumps(self.server.requests[0])
        self.assertIn("PAST_USER",wire);self.assertIn("PAST_ASSISTANT",wire);self.assertIn("CURRENT_USER",wire)
        self.assertEqual(self.server.tool_names,[])
        self.assertTrue(self.session.isolation_verified)

    def test_parallel_mcp_handoff_and_real_id_continuation(self):
        self.server.mode="tools"
        tool={"type":"function","function":{"name":"clock","description":"Fixture clock","parameters":{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"]}}}
        messages=[{"role":"system","content":"Fixture persona."},{"role":"user","content":"Check two zones."}]
        first=self.request(messages,[tool])["message"]
        self.assertEqual(len(first["tool_calls"]),2)
        self.assertEqual(set(self.server.tool_names),{"mcp__atlas__clock"})
        messages.append(first)
        for call in first["tool_calls"]:
            messages.append({"role":"tool","name":"clock","tool_call_id":call["id"],"content":"FIXTURE_RESULT_"+call["function"]["arguments"]["zone"]})
        self.server.mode="text"
        final=self.request(messages)["message"]["content"]
        self.assertEqual(final,"A clear fixture reply.")
        self.assertEqual(len(self.server.requests),2)
        serialized=json.dumps(self.server.requests[-1])
        for call in first["tool_calls"]:self.assertIn(call["id"],serialized)
        self.assertIn("FIXTURE_RESULT_Asia/Tokyo",serialized)

    def test_rate_limit_stops_before_automatic_retry(self):
        self.server.mode="limited"
        with self.assertRaises(ProviderError) as caught:self.request([{"role":"user","content":"Fixture question"}])
        self.assertEqual(caught.exception.code,"usage_limit")
        self.assertEqual(len(self.server.requests),1)
        self.assertTrue(self.session.status()["usage_paused"])

    def test_real_image_block_reaches_cli_model_request(self):
        image="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jZ1kAAAAASUVORK5CYII="
        result=self.request([{"role":"user","content":"Describe the attached fixture.","images":[image]}])
        self.assertEqual(result["message"]["content"],"A clear fixture reply.")
        self.assertEqual(len(self.server.requests),1)
        messages=self.server.requests[0]["messages"]
        blocks=[block for message in messages for block in message["content"] if isinstance(block,dict)]
        images=[block for block in blocks if block.get("type")=="image"]
        self.assertEqual(len(images),1)
        self.assertEqual(images[0]["source"]["data"],image)

    def test_cancel_stream_reaps_owned_cli_and_next_turn_works(self):
        import claude_native
        self.server.mode="slow"
        response=self.session.post("/api/chat",json={"_atlas_request_id":"cancel-fixture","messages":[{"role":"user","content":"Fixture question"}]})
        stream=response.iter_content()
        stopped=[]
        real_stop=claude_native.stop_process
        def stop(proc):
            real_stop(proc);stopped.append(proc)
        with patch("claude_native.stop_process",side_effect=stop):
            self.assertIn(b"fixture",next(stream))
            start=time.monotonic()
            self.session.interrupt("cancel-fixture")
            stream.close()
            self.assertLess(time.monotonic()-start,3)
        self.assertTrue(stopped)
        self.assertTrue(all(proc.poll() is not None for proc in stopped))
        self.server.release.set();self.server.mode="text"
        self.assertEqual(self.request([{"role":"user","content":"Next fixture turn"}])["message"]["content"],"A clear fixture reply.")


if __name__=="__main__":unittest.main()
